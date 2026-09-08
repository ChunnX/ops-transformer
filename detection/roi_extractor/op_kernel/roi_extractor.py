# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details on using this file.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""RoiExtractor kernel (pure Vector, single section_vector) — Stage 5 optimized.

FPN RoI Extract (RoIAlignV2 semantics, SPEC.md machine contract):
per ROI select an FPN level by fp16(sqrt_fp32(area)), map coordinates with
spatial_scale (aligned: -0.5), bilinear-sample sh*sw points per output pixel,
average and write y[y_row, c, ph, pw] (y_row = index[n] when index is given).

Stage 5 restructure vs Stage 4 (PERFORMANCE_REPORT.md §5.2, precision-preserving:
identical per-pixel (i-major, j-minor) fp32 accumulation order and formulas):

1. Strip wide view [896, 48] with a runtime per-channel pitch (ys_c): the ROI's
   whole feature window is staged ONCE per (channel-block, row-group) instead of
   once per pixel — MTE2 slice-load count drops by up to 49x. A per-pixel-chunked
   fallback path covers windows wider than 48 columns / taller than ys_cap.
2. Sample coordinates stay in the kernel scalar domain but each fp32->int32
   FLOOR round-trip batches four x values through the [1, 8] coord tile (one
   TCVT per 4 samples), the y-side value is cast once per (pixel row, i) and
   reused across the j loop, and int->float promotions are spelled directly
   (the Stage 4 sw_f/pooled_w_f/ph_f/pw_f cast round-trips are gone).
3. The (j+0.5)*inv_sw fraction tables are removed: the expression is two scalar
   ALU ops, cheaper than one masked UB getval.
4. The pixel loop is nested (ph rows outer, pw columns inner, per-row stores)
   removing the uint64 px/pooled_w division pair per pixel.
5. Four samples share one _bilinear_acc4_vf call: the acc tile round-trip and
   lane-base setup amortize over 4 samples (general-12/19).

Work item = (roi n, channel block cb). Stage 4 contract otherwise unchanged:
8-slot padded fixed-arity TensorList ABI, single launch, auto_mutex tile
groups, VF for all vector numerics, fp32 coordinate/weight/accumulator chain,
single RINT on the fp16 output.
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.tilingkey import TilingKeyField


# ================================================================
#  Tile dimensions and constants
# ================================================================
CH_BLOCK = 64     # channels per work item = fp32 VF register lane width
STRIP_COLS = 48   # strip physical columns (fp32: 48*4B = 192B aligned)
STRIP_ROWS = 896  # strip physical rows (64ch x 14 at full channels)
STRIP_ELEMS = STRIP_ROWS * STRIP_COLS   # 43008 strip elements (fp32)
STAGE_ROWS = 56   # staging physical rows (fp16 hop; >= any ys_cap)
STAGE_COLS = 48   # staging physical columns (fp16: 48*2B = 96B aligned)
PX_PHYS = 64      # out_tile pixel columns per stored group
FRAC_N = 128      # legacy sample-count bound kept for parity notes
XS_BUDGET = 46    # fallback per-pixel x-chunk margin (2 cols for hi taps)
JB = 4            # samples batched per VF accumulation call

# ================================================================
#  UB(Vec) address map (platform ini key ub_size=253952)
#  High water: staging [56,48] in the fp32 datatype key is the worst case:
#  0x2F000 + 56*48*4 = 0x31A00 = 203264 / 253952 (80.0%).
# ================================================================
VA_STRIP = 0x00000     # [896,48]  FP32  strip residency            172032B
VA_OUT = 0x2A000       # [64,64]   FP32  out_tile (+fp16 view)       16384B
VA_ROIS_RAW = 0x2EE00  # [1,16]    input_dtype rois row                 64B
VA_ROIS_F32 = 0x2EE40  # [1,8]     FP32  rois row widened                32B
VA_COORD_A = 0x2EE60   # [1,8]     FP32  scalar cast source             32B
VA_COORD_I = 0x2EE80   # [1,8]     INT32 scalar cast result             32B
VA_COORD_F16 = 0x2EEA0 # [1,16]    FP16  level-chain hop                32B
VA_ACC = 0x2EEC0       # [1,64]    FP32  pixel accumulator              256B
VA_STAGING = 0x2F000   # [56,48]   input_dtype fp16 load stage         5376B
# 0x2E000..0x2EE00 (3584B) retired: the old [28,32] staging region.


# ================================================================
#  Tiling data (must match op_host/roi_extractor_tiling.h exactly)
# ================================================================
@dataclass
class RoiExtractorTiling:
    r: int               # R (rois rows)
    n: int               # N (batch)
    c: int               # C (channels)
    l: int               # valid level count 1..8 (features prefix length)
    pooled_h: int        # attr pooled_height
    pooled_w: int        # attr pooled_width (independent of pooled_h)
    sample_num: int      # attr sample_num (0 = adaptive ceil(bin))
    finest_scale: int    # attr finest_scale
    aligned: int         # attr aligned bool -> 0/1
    is_fp16: int         # features dtype == fp16 ? 1 : 0
    feat_h: "int[8]"     # per-level H_l; absent levels (l>=L) padded with 1
    feat_w: "int[8]"     # per-level W_l; same padding
    spatial_scale: "float[8]"  # attr spatial_scale, first L entries valid
    b_pow2: "float[9]"   # B[i] = 2^i * finest_scale, i=0..8 (fp16-exact fp32)


# ================================================================
#  Tiling key (HasIndex only, values=[0,1])
# ================================================================
class RoiExtractorTilingKey:
    HasIndex = TilingKeyField(bits=1, values=[0, 1])


# ================================================================
#  VF helpers (masks are always created inside the VF body: passing a
#  create_mask/update_mask result inline as a call-site argument makes
#  bisheng crash in instruction selection — verified on target)
# ================================================================


@pl.vector_function
def _sqrt_lane0_vf(coord_a):
    """S3: fp32 sqrt (precision=True) of element 0, in place."""
    preg = vf.update_mask(1, dtype=pl.DT_FP32)
    reg = vf.load_align(coord_a, 0)
    reg = vf.sqrt(reg, preg, precision=True)
    vf.store_align(coord_a, reg, preg)


@pl.vector_function
def _zero_acc_vf(acc_tile):
    """Pixel head: acc = 0 on all 64 channel lanes."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zero = vf.full(0.0, dtype=pl.DT_FP32)
    vf.store_align(acc_tile, zero, preg)


@pl.vector_function
def _bilinear_acc_vf(strip_tile, acc_tile, pitch, c_eff,
                     pos00, pos01, pos10, pos11, w00, w01, w10, w11):
    """One sample: 4-point gather + weighted fp32 accumulate."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_ch = vf.update_mask(c_eff, dtype=pl.DT_FP32)
    lane_base = vf.arange(0, dtype=pl.DT_UINT32)
    lane_base = vf.muls(lane_base, pitch, preg)

    off00 = vf.adds(lane_base, pos00, preg)
    off00 = vf.mins(off00, STRIP_ELEMS - 1, preg)
    v00 = vf.gather(strip_tile, off00, preg_ch)
    off01 = vf.adds(lane_base, pos01, preg)
    off01 = vf.mins(off01, STRIP_ELEMS - 1, preg)
    v01 = vf.gather(strip_tile, off01, preg_ch)
    off10 = vf.adds(lane_base, pos10, preg)
    off10 = vf.mins(off10, STRIP_ELEMS - 1, preg)
    v10 = vf.gather(strip_tile, off10, preg_ch)
    off11 = vf.adds(lane_base, pos11, preg)
    off11 = vf.mins(off11, STRIP_ELEMS - 1, preg)
    v11 = vf.gather(strip_tile, off11, preg_ch)

    acc = vf.load_align(acc_tile, 0)
    t00 = vf.muls(v00, w00, preg_ch)
    acc = vf.add(acc, t00, preg_ch)
    t01 = vf.muls(v01, w01, preg_ch)
    acc = vf.add(acc, t01, preg_ch)
    t10 = vf.muls(v10, w10, preg_ch)
    acc = vf.add(acc, t10, preg_ch)
    t11 = vf.muls(v11, w11, preg_ch)
    acc = vf.add(acc, t11, preg_ch)
    vf.store_align(acc_tile, acc, preg)


@pl.vector_function
def _bilinear_acc4_vf(strip_tile, acc_tile, pitch, c_eff,
                      a00, a01, a10, a11, b00, b01, b10, b11,
                      c00, c01, c10, c11, d00, d01, d10, d11,
                      e00, e01, e10, e11, f00, f01, f10, f11,
                      g00, g01, g10, g11, h00, h01, h10, h11):
    """Four samples (a..h = per-sample pos00..pos11/w00..w11 pairs).

    Layout: (a, b) = sample 1 (pos, w); (c, d) = sample 2; (e, f) = sample 3;
    (g, h) = sample 4. Each letter carries pos00, pos01, pos10, pos11 or
    w00, w01, w10, w11 respectively. acc round-trips once for all four.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_ch = vf.update_mask(c_eff, dtype=pl.DT_FP32)
    lane_base = vf.arange(0, dtype=pl.DT_UINT32)
    lane_base = vf.muls(lane_base, pitch, preg)

    acc = vf.load_align(acc_tile, 0)
    # ---- sample 1 (a = pos, b = w) ----
    o = vf.adds(lane_base, a00, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, b00, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, a01, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, b01, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, a10, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, b10, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, a11, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, b11, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    # ---- sample 2 (c, d) ----
    o = vf.adds(lane_base, c00, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, d00, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, c01, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, d01, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, c10, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, d10, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, c11, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, d11, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    # ---- sample 3 (e, f) ----
    o = vf.adds(lane_base, e00, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, f00, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, e01, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, f01, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, e10, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, f10, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, e11, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, f11, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    # ---- sample 4 (g, h) ----
    o = vf.adds(lane_base, g00, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, h00, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, g01, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, h01, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, g10, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, h10, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    o = vf.adds(lane_base, g11, preg)
    o = vf.mins(o, STRIP_ELEMS - 1, preg)
    v = vf.gather(strip_tile, o, preg_ch)
    t = vf.muls(v, h11, preg_ch)
    acc = vf.add(acc, t, preg_ch)
    vf.store_align(acc_tile, acc, preg)


@pl.vector_function
def _finish_pixel_vf(acc_tile, out_tile, inv_area, px_off, c_eff):
    """S14: out column px_off = acc * inv_area, scattered per channel lane."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_ch = vf.update_mask(c_eff, dtype=pl.DT_FP32)
    acc = vf.load_align(acc_tile, 0)
    out = vf.muls(acc, inv_area, preg_ch)
    idx = vf.arange(0, dtype=pl.DT_UINT32)
    idx = vf.muls(idx, PX_PHYS, preg)
    idx = vf.adds(idx, px_off, preg)
    vf.scatter(out_tile, out, idx, preg_ch)


# ================================================================
#  Kernel — dynamic rank: inputs are raw pointers, shapes from tiling
# ================================================================
def roi_extractor_inner(
    features_0: pl.Ptr[pl.DT_UINT8],
    features_1: pl.Ptr[pl.DT_UINT8],
    features_2: pl.Ptr[pl.DT_UINT8],
    features_3: pl.Ptr[pl.DT_UINT8],
    features_4: pl.Ptr[pl.DT_UINT8],
    features_5: pl.Ptr[pl.DT_UINT8],
    features_6: pl.Ptr[pl.DT_UINT8],
    features_7: pl.Ptr[pl.DT_UINT8],
    rois: pl.Ptr[pl.DT_UINT8],
    index: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8],
    tiling: RoiExtractorTiling,
):
    rois_view = pl.make_tensor(rois, [tiling.r, 5], [5, 1], dtype=input_dtype)
    idx_view = pl.make_tensor(index, [tiling.r, 1], [1, 1], dtype=pl.DT_INT32)
    n_pxw = tiling.pooled_h * tiling.pooled_w
    y_view = pl.make_tensor(y, [tiling.r * tiling.c, n_pxw], [n_pxw, 1],
                            dtype=input_dtype)

    # ---- tile groups (declared ONCE above the section; FA-05) ----
    strip_g = pl.make_tile_group(
        type=pl.TileType(shape=[STRIP_ROWS, STRIP_COLS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=VA_STRIP, mutex_ids=[0])
    out_g = pl.make_tile_group(
        type=pl.TileType(shape=[PX_PHYS, PX_PHYS], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=VA_OUT, mutex_ids=[1])
    # fp16 view of the same UB region (in-place fp32->fp16 RINT at S15);
    # shares mutex id 1 with the fp32 group (same-address serialisation).
    out16_g = pl.make_tile_group(
        type=pl.TileType(shape=[PX_PHYS, PX_PHYS], dtype=input_dtype,
                         target_memory=pl.MemorySpace.Vec),
        addrs=VA_OUT, mutex_ids=[1])
    stage_g = pl.make_tile_group(
        type=pl.TileType(shape=[STAGE_ROWS, STAGE_COLS], dtype=input_dtype,
                         target_memory=pl.MemorySpace.Vec),
        addrs=VA_STAGING, mutex_ids=[2])
    roisraw_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, 16], dtype=input_dtype,
                         target_memory=pl.MemorySpace.Vec),
        addrs=VA_ROIS_RAW, mutex_ids=[3])
    roisf32_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=VA_ROIS_F32, mutex_ids=[4])
    coorda_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=VA_COORD_A, mutex_ids=[5])
    coordi_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_INT32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=VA_COORD_I, mutex_ids=[7])
    coordf16_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, 16], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=VA_COORD_F16, mutex_ids=[8])
    acc_g = pl.make_tile_group(
        type=pl.TileType(shape=[1, CH_BLOCK], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=VA_ACC, mutex_ids=[11])

    with pl.section_vector():
        core_id = pl.get_block_idx()
        num_cores = pl.get_block_num()
        n_cb = (tiling.c + CH_BLOCK - 1) // CH_BLOCK
        total = tiling.r * n_cb

        for wid in pl.range(core_id, total, num_cores):
            n = wid // n_cb
            cb = wid % n_cb
            c0 = cb * CH_BLOCK
            c_eff = pl.min(tiling.c - c0, CH_BLOCK)

            # ---- S1: rois row -> fp32 scalars ----
            rois_f32 = roisf32_g.current()
            if input_dtype == pl.DT_FP16:
                rois_raw = roisraw_g.current()
                pl.set_validshape(rois_raw, [1, 5])
                pl.load(rois_raw, rois_view, [n, 0])
                pl.cast(rois_f32, rois_raw, mode=pl.RoundMode.CAST_NONE)
            else:
                pl.set_validshape(rois_f32, [1, 5])
                pl.load(rois_f32, rois_view, [n, 0])
            bidx_r = pl.getval(rois_f32, 0)
            x0r = pl.getval(rois_f32, 1)
            y0r = pl.getval(rois_f32, 2)
            x1r = pl.getval(rois_f32, 3)
            y1r = pl.getval(rois_f32, 4)

            # ---- S2: batch guard ----
            coord_a = coorda_g.current()
            coord_i = coordi_g.current()
            coord_f16 = coordf16_g.current()
            coord_a[0, 0] = bidx_r
            pl.cast(coord_i, coord_a, mode=pl.RoundMode.CAST_FLOOR)
            b = pl.getval(coord_i, 0)
            if b < 0 or b >= tiling.n:
                continue

            # ---- S3: level selection (fp32 sqrt -> fp16 compare) ----
            prod = (x1r - x0r) * (y1r - y0r)
            if prod < 0.0:
                continue
            coord_a[0, 0] = prod
            _sqrt_lane0_vf(coord_a)
            pl.set_validshape(coord_f16, [1, 8])
            pl.cast(coord_f16, coord_a, mode=pl.RoundMode.CAST_RINT)
            pl.set_validshape(rois_f32, [1, 8])
            pl.cast(rois_f32, coord_f16, mode=pl.RoundMode.CAST_NONE)
            s16 = pl.getval(rois_f32, 0)

            lvl = 0
            if s16 <= tiling.b_pow2[1]:
                lvl = 0
            elif tiling.l == 1:
                continue
            elif s16 > tiling.b_pow2[tiling.l - 1]:
                lvl = tiling.l - 1
            else:
                lvl = tiling.l - 1
                for li in pl.range(1, tiling.l - 1):
                    if s16 > tiling.b_pow2[li] and s16 <= tiling.b_pow2[li + 1]:
                        lvl = li

            # ---- S4: coordinate mapping (fp32 chain; direct promotions) ----
            sc = tiling.spatial_scale[lvl]
            x0 = x0r * sc
            y0 = y0r * sc
            x1 = x1r * sc
            y1 = y1r * sc
            if tiling.aligned == 1:
                x0 = x0 - 0.5
                y0 = y0 - 0.5
                x1 = x1 - 0.5
                y1 = y1 - 0.5
            roi_w = max(x1 - x0, 1.0)
            roi_h = max(y1 - y0, 1.0)
            bin_w = roi_w / (tiling.pooled_w + 0.0)
            bin_h = roi_h / (tiling.pooled_h + 0.0)

            # ---- S5: sampling counts ----
            if tiling.sample_num > 0:
                sw = tiling.sample_num
                sh = tiling.sample_num
            else:
                coord_a[0, 0] = bin_w
                coord_a[0, 1] = bin_h
                pl.cast(coord_i, coord_a, mode=pl.RoundMode.CAST_CEIL)
                sw = pl.getval(coord_i, 0)
                sh = pl.getval(coord_i, 1)
                if sw < 1:
                    sw = 1
                if sh < 1:
                    sh = 1
            inv_sw = 1.0 / (sw + 0.0)
            inv_sh = 1.0 / (sh + 0.0)
            inv_area = 1.0 / ((sw + 0.0) * (sh + 0.0))

            # ---- S6: output row mapping ----
            if HasIndex == 1:
                y_row = pl.getval(idx_view, n)
            else:
                y_row = n

            # ---- S7: level view / dims selection ----
            feat_ptr = features_0
            lvl_h = tiling.feat_h[0]
            lvl_w = tiling.feat_w[0]
            if lvl == 1:
                feat_ptr = features_1
                lvl_h = tiling.feat_h[1]
                lvl_w = tiling.feat_w[1]
            if lvl == 2:
                feat_ptr = features_2
                lvl_h = tiling.feat_h[2]
                lvl_w = tiling.feat_w[2]
            if lvl == 3:
                feat_ptr = features_3
                lvl_h = tiling.feat_h[3]
                lvl_w = tiling.feat_w[3]
            if lvl == 4:
                feat_ptr = features_4
                lvl_h = tiling.feat_h[4]
                lvl_w = tiling.feat_w[4]
            if lvl == 5:
                feat_ptr = features_5
                lvl_h = tiling.feat_h[5]
                lvl_w = tiling.feat_w[5]
            if lvl == 6:
                feat_ptr = features_6
                lvl_h = tiling.feat_h[6]
                lvl_w = tiling.feat_w[6]
            if lvl == 7:
                feat_ptr = features_7
                lvl_h = tiling.feat_h[7]
                lvl_w = tiling.feat_w[7]
            lvl_view = pl.make_tensor(
                feat_ptr,
                [tiling.n * tiling.c * tiling.feat_h[lvl], tiling.feat_w[lvl]],
                [tiling.feat_w[lvl], 1],
                dtype=input_dtype)
            lvl_w_m1 = (lvl_w - 1) + 0.0
            lvl_h_m1 = (lvl_h - 1) + 0.0
            lvl_w_f = lvl_w + 0.0
            lvl_h_f = lvl_h + 0.0

            strip = strip_g.current()
            acc = acc_g.current()
            out = out_g.current()
            stage = stage_g.current()

            # ---- ROI-level x window (fast-path shared staging) ----
            # NB: the window edges replicate the per-sample expressions
            # bit-identically (same association: xbase + (t*inv_sw)*bin_w),
            # so floor(window_edge) can never round below the extreme
            # sample's x_lo and leave its hi tap outside the loaded window.
            xf = (x0 + 0 * bin_w) + (0.5 * inv_sw) * bin_w
            xl = (x0 + (tiling.pooled_w - 1) * bin_w) \
                + ((sw - 1 + 0.5) * inv_sw) * bin_w
            xf_c = min(max(xf, 0.0), lvl_w_m1)
            xl_c = min(max(xl, 0.0), lvl_w_m1)
            coord_a[0, 0] = xf_c
            coord_a[0, 1] = xl_c
            pl.cast(coord_i, coord_a, mode=pl.RoundMode.CAST_FLOOR)
            xc0 = pl.getval(coord_i, 0)
            xl_fl = pl.getval(coord_i, 1)
            x_end = pl.min(xl_fl + 1, lvl_w - 1)
            xs_roi = x_end - xc0 + 1

            # ---- row budget: strip rows / staging rows ----
            ys_cap = pl.min(STAGE_ROWS, STRIP_ROWS // c_eff)

            use_fast = 1
            if xs_roi > STRIP_COLS:
                use_fast = 0
            if bin_h + 2.0 > ys_cap + 0.0:
                use_fast = 0

            if use_fast == 1:
                # rows per shared chunk: K*bin_h + 2 <= ys_cap
                coord_a[0, 0] = (ys_cap - 2) / bin_h
                pl.cast(coord_i, coord_a, mode=pl.RoundMode.CAST_FLOOR)
                k_raw = pl.getval(coord_i, 0)
                k_rows = pl.max(pl.min(k_raw, tiling.pooled_h), 1)

                for ph0 in pl.range(0, tiling.pooled_h, k_rows):
                    ph1 = pl.min(ph0 + k_rows, tiling.pooled_h)
                    # chunk y window from the group's first/last samples
                    # (bit-identical association to the per-sample y chain)
                    yf = (y0 + ph0 * bin_h) + (0.5 * inv_sh) * bin_h
                    yl = (y0 + (ph1 - 1) * bin_h) \
                        + ((sh - 1 + 0.5) * inv_sh) * bin_h
                    yf_c = min(max(yf, 0.0), lvl_h_m1)
                    yl_c = min(max(yl, 0.0), lvl_h_m1)
                    coord_a[0, 0] = yf_c
                    coord_a[0, 1] = yl_c
                    pl.cast(coord_i, coord_a, mode=pl.RoundMode.CAST_FLOOR)
                    yc0 = pl.getval(coord_i, 0)
                    yl_fl = pl.getval(coord_i, 1)
                    y_end = pl.min(yl_fl + 1, lvl_h - 1)
                    ys_c = y_end - yc0 + 1
                    pitch = ys_c * STRIP_COLS

                    # ---- S11: strip load (per channel), whole-row-group window ----
                    pl.system.bar_mte2()
                    for c in pl.range(0, c_eff):
                        sl = strip[c * ys_c:(c + 1) * ys_c, 0:xs_roi]
                        pl.set_validshape(sl, [ys_c, xs_roi])
                        row_off = b * tiling.c * lvl_h + (c0 + c) * lvl_h + yc0
                        if input_dtype == pl.DT_FP16:
                            pl.set_validshape(stage, [ys_c, xs_roi])
                            pl.load(stage, lvl_view, [row_off, xc0])
                            pl.cast(sl, stage, mode=pl.RoundMode.CAST_NONE)
                        else:
                            pl.load(sl, lvl_view, [row_off, xc0])
                    pl.set_validshape(strip, [STRIP_ROWS, STRIP_COLS])

                    # ---- pixels of the row group (per-row stores) ----
                    for ph in pl.range(ph0, ph1):
                        ybase = y0 + ph * bin_h
                        for pw0 in pl.range(0, tiling.pooled_w, PX_PHYS):
                            n_cols = pl.min(PX_PHYS, tiling.pooled_w - pw0)
                            for pw in pl.range(pw0, pw0 + n_cols):
                                xbase = x0 + pw * bin_w
                                _zero_acc_vf(acc)
                                for i in pl.range(0, sh):
                                    yc = ybase + (i + 0.5) * inv_sh * bin_h
                                    if yc < -1.0 or yc > lvl_h_f:
                                        continue
                                    yc_c = min(max(yc, 0.0), lvl_h_m1)
                                    coord_a[0, 0] = yc_c
                                    pl.cast(coord_i, coord_a,
                                            mode=pl.RoundMode.CAST_FLOOR)
                                    y_lo = pl.getval(coord_i, 0)
                                    y_hi = pl.min(y_lo + 1, lvl_h - 1)
                                    ly = yc_c - y_lo
                                    hy = 1.0 - ly
                                    py0 = (y_lo - yc0) * STRIP_COLS
                                    py1 = (y_hi - yc0) * STRIP_COLS
                                    nj4 = (sw // JB) * JB
                                    for j4 in pl.range(0, nj4, JB):
                                        xa = xbase + (j4 + 0.5) * inv_sw * bin_w
                                        xb = xbase + (j4 + 1.5) * inv_sw * bin_w
                                        xc = xbase + (j4 + 2.5) * inv_sw * bin_w
                                        xd = xbase + (j4 + 3.5) * inv_sw * bin_w
                                        xa_c = min(max(xa, 0.0), lvl_w_m1)
                                        xb_c = min(max(xb, 0.0), lvl_w_m1)
                                        xc_c = min(max(xc, 0.0), lvl_w_m1)
                                        xd_c = min(max(xd, 0.0), lvl_w_m1)
                                        coord_a[0, 0] = xa_c
                                        coord_a[0, 1] = xb_c
                                        coord_a[0, 2] = xc_c
                                        coord_a[0, 3] = xd_c
                                        pl.cast(coord_i, coord_a,
                                                mode=pl.RoundMode.CAST_FLOOR)
                                        xl_a = pl.getval(coord_i, 0)
                                        xl_b = pl.getval(coord_i, 1)
                                        xl_c2 = pl.getval(coord_i, 2)
                                        xl_d = pl.getval(coord_i, 3)
                                        # OOB samples contribute 0 (SPEC: out-of-
                                        # bounds samples count in the denominator
                                        # only). Their gather addresses are pinned
                                        # to offset 0 (inside the loaded window:
                                        # finite, so weight*value = 0 exactly);
                                        # the clamped coordinate itself may fall
                                        # outside the loaded window or below the
                                        # strip base, which must not be gathered.
                                        pa0 = 0
                                        pa1 = 0
                                        pa2 = 0
                                        pa3 = 0
                                        wa0 = 0.0
                                        wa1 = 0.0
                                        oob_a = 0
                                        if xa < -1.0 or xa > lvl_w_f:
                                            oob_a = 1
                                        if oob_a == 0:
                                            lxa = xa_c - xl_a
                                            wa1 = lxa
                                            wa0 = 1.0 - lxa
                                            xh_a = pl.min(xl_a + 1, lvl_w - 1)
                                            pa0 = py0 + (xl_a - xc0)
                                            pa1 = py0 + (xh_a - xc0)
                                            pa2 = py1 + (xl_a - xc0)
                                            pa3 = py1 + (xh_a - xc0)
                                        pb0 = 0
                                        pb1 = 0
                                        pb2 = 0
                                        pb3 = 0
                                        wb0 = 0.0
                                        wb1 = 0.0
                                        oob_b = 0
                                        if xb < -1.0 or xb > lvl_w_f:
                                            oob_b = 1
                                        if oob_b == 0:
                                            lxb = xb_c - xl_b
                                            wb1 = lxb
                                            wb0 = 1.0 - lxb
                                            xh_b = pl.min(xl_b + 1, lvl_w - 1)
                                            pb0 = py0 + (xl_b - xc0)
                                            pb1 = py0 + (xh_b - xc0)
                                            pb2 = py1 + (xl_b - xc0)
                                            pb3 = py1 + (xh_b - xc0)
                                        pc0 = 0
                                        pc1 = 0
                                        pc2 = 0
                                        pc3 = 0
                                        wc0 = 0.0
                                        wc1 = 0.0
                                        oob_c = 0
                                        if xc < -1.0 or xc > lvl_w_f:
                                            oob_c = 1
                                        if oob_c == 0:
                                            lxc = xc_c - xl_c2
                                            wc1 = lxc
                                            wc0 = 1.0 - lxc
                                            xh_c = pl.min(xl_c2 + 1, lvl_w - 1)
                                            pc0 = py0 + (xl_c2 - xc0)
                                            pc1 = py0 + (xh_c - xc0)
                                            pc2 = py1 + (xl_c2 - xc0)
                                            pc3 = py1 + (xh_c - xc0)
                                        pd0 = 0
                                        pd1 = 0
                                        pd2 = 0
                                        pd3 = 0
                                        wd0 = 0.0
                                        wd1 = 0.0
                                        oob_d = 0
                                        if xd < -1.0 or xd > lvl_w_f:
                                            oob_d = 1
                                        if oob_d == 0:
                                            lxd = xd_c - xl_d
                                            wd1 = lxd
                                            wd0 = 1.0 - lxd
                                            xh_d = pl.min(xl_d + 1, lvl_w - 1)
                                            pd0 = py0 + (xl_d - xc0)
                                            pd1 = py0 + (xh_d - xc0)
                                            pd2 = py1 + (xl_d - xc0)
                                            pd3 = py1 + (xh_d - xc0)
                                        _bilinear_acc4_vf(
                                            strip, acc, pitch, c_eff,
                                            pa0, pa1, pa2, pa3,
                                            hy * wa0, hy * wa1,
                                            ly * wa0, ly * wa1,
                                            pb0, pb1, pb2, pb3,
                                            hy * wb0, hy * wb1,
                                            ly * wb0, ly * wb1,
                                            pc0, pc1, pc2, pc3,
                                            hy * wc0, hy * wc1,
                                            ly * wc0, ly * wc1,
                                            pd0, pd1, pd2, pd3,
                                            hy * wd0, hy * wd1,
                                            ly * wd0, ly * wd1)
                                    for j in pl.range(nj4, sw):
                                        x = xbase + (j + 0.5) * inv_sw * bin_w
                                        if x < -1.0 or x > lvl_w_f:
                                            continue
                                        xc_s = min(max(x, 0.0), lvl_w_m1)
                                        coord_a[0, 0] = xc_s
                                        pl.cast(coord_i, coord_a,
                                                mode=pl.RoundMode.CAST_FLOOR)
                                        x_lo = pl.getval(coord_i, 0)
                                        x_hi = pl.min(x_lo + 1, lvl_w - 1)
                                        lx = xc_s - x_lo
                                        hx = 1.0 - lx
                                        _bilinear_acc_vf(
                                            strip, acc, pitch, c_eff,
                                            py0 + (x_lo - xc0),
                                            py0 + (x_hi - xc0),
                                            py1 + (x_lo - xc0),
                                            py1 + (x_hi - xc0),
                                            hy * hx, hy * lx,
                                            ly * hx, ly * lx)
                                _finish_pixel_vf(acc, out, inv_area,
                                                 pw - pw0, c_eff)
                            # ---- S15: per-row group tail cast + store ----
                            pl.set_validshape(out, [c_eff, n_cols])
                            if input_dtype == pl.DT_FP16:
                                out16 = out16_g.current()
                                pl.set_validshape(out16, [c_eff, n_cols])
                                pl.cast(out16, out, mode=pl.RoundMode.CAST_RINT)
                                pl.store(y_view, out16,
                                         [y_row * tiling.c + c0,
                                          ph * tiling.pooled_w + pw0])
                            else:
                                pl.store(y_view, out,
                                         [y_row * tiling.c + c0,
                                          ph * tiling.pooled_w + pw0])
            else:
                # ---- fallback: per-pixel chunked windows (legacy geometry,
                #      adapted to the wide strip view + runtime pitch) ----
                coord_a[0, 0] = (ys_cap - 2) / bin_h
                pl.cast(coord_i, coord_a, mode=pl.RoundMode.CAST_FLOOR)
                si = pl.getval(coord_i, 0)
                if si < 1:
                    si = 1
                coord_a[0, 0] = XS_BUDGET / bin_w
                pl.cast(coord_i, coord_a, mode=pl.RoundMode.CAST_FLOOR)
                sj = pl.getval(coord_i, 0)
                if sj < 1:
                    sj = 1

                for ph in pl.range(0, tiling.pooled_h):
                    ybase = y0 + ph * bin_h
                    for pw0 in pl.range(0, tiling.pooled_w, PX_PHYS):
                        n_cols = pl.min(PX_PHYS, tiling.pooled_w - pw0)
                        for pw in pl.range(pw0, pw0 + n_cols):
                            xbase = x0 + pw * bin_w
                            _zero_acc_vf(acc)
                            for i0 in pl.range(0, sh, si):
                                i_end = pl.min(i0 + si, sh)
                                for j0 in pl.range(0, sw, sj):
                                    j_end = pl.min(j0 + sj, sw)
                                    fj0 = (j0 + 0.5) * inv_sw
                                    fj1 = (j_end - 1 + 0.5) * inv_sw
                                    fi0 = (i0 + 0.5) * inv_sh
                                    fi1 = (i_end - 1 + 0.5) * inv_sh
                                    xf_c2 = min(max(xbase + fj0 * bin_w, 0.0),
                                                lvl_w_m1)
                                    xl_c3 = min(max(xbase + fj1 * bin_w, 0.0),
                                                lvl_w_m1)
                                    yf_c2 = min(max(ybase + fi0 * bin_h, 0.0),
                                                lvl_h_m1)
                                    yl_c2 = min(max(ybase + fi1 * bin_h, 0.0),
                                                lvl_h_m1)
                                    coord_a[0, 0] = xf_c2
                                    coord_a[0, 1] = xl_c3
                                    coord_a[0, 2] = yf_c2
                                    coord_a[0, 3] = yl_c2
                                    pl.cast(coord_i, coord_a,
                                            mode=pl.RoundMode.CAST_FLOOR)
                                    xc0 = pl.getval(coord_i, 0)
                                    xl_fl = pl.getval(coord_i, 1)
                                    yc0 = pl.getval(coord_i, 2)
                                    yl_fl = pl.getval(coord_i, 3)
                                    xc_hi = pl.min(xl_fl + 1, lvl_w - 1)
                                    yc_hi = pl.min(yl_fl + 1, lvl_h - 1)
                                    xs = xc_hi - xc0 + 1
                                    ys = yc_hi - yc0 + 1
                                    pitch = ys * STRIP_COLS

                                    pl.system.bar_mte2()
                                    for c in pl.range(0, c_eff):
                                        sl = strip[c * ys:(c + 1) * ys, 0:xs]
                                        pl.set_validshape(sl, [ys, xs])
                                        row_off = (b * tiling.c * lvl_h
                                                   + (c0 + c) * lvl_h + yc0)
                                        if input_dtype == pl.DT_FP16:
                                            pl.set_validshape(stage, [ys, xs])
                                            pl.load(stage, lvl_view,
                                                    [row_off, xc0])
                                            pl.cast(sl, stage,
                                                    mode=pl.RoundMode.CAST_NONE)
                                        else:
                                            pl.load(sl, lvl_view,
                                                    [row_off, xc0])
                                    pl.set_validshape(strip,
                                                      [STRIP_ROWS, STRIP_COLS])

                                    for i in pl.range(i0, i_end):
                                        yc = ybase + (i + 0.5) * inv_sh * bin_h
                                        if yc < -1.0 or yc > lvl_h_f:
                                            continue
                                        yc_c = min(max(yc, 0.0), lvl_h_m1)
                                        coord_a[0, 0] = yc_c
                                        pl.cast(coord_i, coord_a,
                                                mode=pl.RoundMode.CAST_FLOOR)
                                        y_lo = pl.getval(coord_i, 0)
                                        y_hi = pl.min(y_lo + 1, lvl_h - 1)
                                        ly = yc_c - y_lo
                                        hy = 1.0 - ly
                                        py0 = (y_lo - yc0) * STRIP_COLS
                                        py1 = (y_hi - yc0) * STRIP_COLS
                                        for j in pl.range(j0, j_end):
                                            x = (xbase
                                                 + (j + 0.5) * inv_sw * bin_w)
                                            if x < -1.0 or x > lvl_w_f:
                                                continue
                                            xc_s = min(max(x, 0.0), lvl_w_m1)
                                            coord_a[0, 0] = xc_s
                                            pl.cast(coord_i, coord_a,
                                                    mode=pl.RoundMode.CAST_FLOOR)
                                            x_lo = pl.getval(coord_i, 0)
                                            x_hi = pl.min(x_lo + 1, lvl_w - 1)
                                            lx = xc_s - x_lo
                                            hx = 1.0 - lx
                                            _bilinear_acc_vf(
                                                strip, acc, pitch, c_eff,
                                                py0 + (x_lo - xc0),
                                                py0 + (x_hi - xc0),
                                                py1 + (x_lo - xc0),
                                                py1 + (x_hi - xc0),
                                                hy * hx, hy * lx,
                                                ly * hx, ly * lx)
                            _finish_pixel_vf(acc, out, inv_area,
                                             pw - pw0, c_eff)
                        pl.set_validshape(out, [c_eff, n_cols])
                        if input_dtype == pl.DT_FP16:
                            out16 = out16_g.current()
                            pl.set_validshape(out16, [c_eff, n_cols])
                            pl.cast(out16, out, mode=pl.RoundMode.CAST_RINT)
                            pl.store(y_view, out16,
                                     [y_row * tiling.c + c0,
                                      ph * tiling.pooled_w + pw0])
                        else:
                            pl.store(y_view, out,
                                     [y_row * tiling.c + c0,
                                      ph * tiling.pooled_w + pw0])
    return


# Thin @pl.jit binding shell: the implementation lives once in
# roi_extractor_inner above (inlined into the caller's IR by the parser, the
# dense_lightning kernel->inner pattern). The JIT-precision test file
# (custom/roi_extractor/test_roi_extractor.py) binds the same inner function
# with an identical shell so the kernel implementation stays single-sourced.
@pl.jit(
    auto_mutex=True,
    tiling_key=RoiExtractorTilingKey,
    datatype={"rois": "input_dtype"},
)
def roi_extractor(
    features_0: pl.Ptr[pl.DT_UINT8],
    features_1: pl.Ptr[pl.DT_UINT8],
    features_2: pl.Ptr[pl.DT_UINT8],
    features_3: pl.Ptr[pl.DT_UINT8],
    features_4: pl.Ptr[pl.DT_UINT8],
    features_5: pl.Ptr[pl.DT_UINT8],
    features_6: pl.Ptr[pl.DT_UINT8],
    features_7: pl.Ptr[pl.DT_UINT8],
    rois: pl.Ptr[pl.DT_UINT8],
    index: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8],
    tiling: RoiExtractorTiling,
):
    roi_extractor_inner(
        features_0, features_1, features_2, features_3,
        features_4, features_5, features_6, features_7,
        rois, index, y, tiling,
    )
