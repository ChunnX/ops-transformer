# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the LICENSE file in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""RoiExtractor kernel (PyPTO Pro).

Multi-level RoI feature extraction: FPN level mapping + RoIAlign bilinear
interpolation pooling, compatible with mmdetection SingleRoIExtractor and the
legacy GE RoiExtractor attribute surface.

Work item = (roi k, channel block cb). For each output row ph and sample row ih
the two feature rows y0/y1 of the RoI's mapped level are loaded into UB as a
[CB, SPAN] channel-row window anchored at column 0 (order=[1,3] over the
level's NCHW view); every sample point then gathers its four bilinear corners
from UB via vf.gather (index and weights computed in-register with vf.astype)
and accumulates into an [PW_MAX, CB] fp32 accumulator tile. avg divides by
n_h*n_w, max reduces with vf.max. RoIs whose window does not fit SPAN take a
per-corner strided-load fallback.

Scalar float->int conversions use a vf.astype + getval roundtrip over a small
int32 scratch tile; per-sample-point floor/weights live inside the vector
functions (vf ops only, C++-safe).

TilingKey bit0 selects the avg/max binary; dtype is covered by the input_dtype
datatype mapping (one binary per dtype). All remaining semantics (aligned,
sample_num, [K,4]+index, level count L<=5, per-level shapes) are runtime
scalars in RoiExtractorTiling.
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


# ================================================================
#  Compile-time tile geometry
# ================================================================
CB = 64            # channel lanes per work item (one 256B fp32 register)
SPAN = 128         # max columns of one row-load window (anchored at column 0)
PW_MAX = 56        # max pooled width (accumulator tile dim)
NEG_INF = -3.4028234663852886e38

# UB byte addresses
A_T0F = 0            # [CB, SPAN] FP32                32768B
A_T1F = 32768        # [CB, SPAN] FP32                32768B
A_ACC = 65536        # [PW_MAX, CB] FP32              14336B
A_OUTF = 81920       # [CB, 1] FP32                   256B
A_S00F = 82432       # [CB, 1] FP32 fallback corners  4 x 256B
A_S01F = 82688
A_S10F = 82944
A_S11F = 83200
A_INTS = 83456       # [CB, 1] INT32 scratch (floor roundtrip) 256B
A_T0X = 90112        # [CB, SPAN] fp16 staging        16384B
A_T1X = 106496       # [CB, SPAN] fp16 staging        16384B
A_SCRX = 122880      # [CB, 1] input_dtype            256B
A_OUTX = 123392      # [CB, 1] input_dtype            256B
# fp32 variant peak ~= 84KB, fp16 variant peak ~= 124KB (180KB budget)


# ================================================================
#  Tiling data — must stay field-identical with op_host
#  roi_extractor_tiling.h (three-side contract, docs/MDE.md §2.1)
# ================================================================
@dataclass
class RoiExtractorTiling:
    numLevels: int        # L, number of non-empty feats slots [1,5]
    roisNum: int          # K
    batchSize: int        # N (identical across levels, host-verified)
    channels: int         # C
    pooledHeight: int     # PH [1,56]
    pooledWidth: int      # PW [1,56]
    sampleNum: int        # 0 = adaptive ceil(bin), [0,16]
    finestScale: int      # level-mapping base scale, default 56
    roiScaleFactor: float # >0: divide RoI coords before level mapping
    aligned: int          # 0/1
    hasIndex: int         # 0 = rois [K,5], 1 = rois [K,4] + index [K]
    colOffset: int        # first coordinate column of rois (0 or 1)
    spatialScale0: float  # per-level scale (multiply semantics), unused 1.0
    spatialScale1: float
    spatialScale2: float
    spatialScale3: float
    spatialScale4: float
    levelH0: int          # per-level feature height, unused 1
    levelH1: int
    levelH2: int
    levelH3: int
    levelH4: int
    levelW0: int          # per-level feature width, unused 1
    levelW1: int
    levelW2: int
    levelW3: int
    levelW4: int


# ================================================================
#  TilingKey — bit layout must match op_host tiling.cpp
#  SetTilingKey (three-side contract, docs/MDE.md §2.2)
# ================================================================
class RoiExtractorTilingKey:
    pool_mode = TilingKeyField(bits=1, values=[0, 1])  # bit0: 0=avg, 1=max


# ================================================================
#  VF helpers
# ================================================================


@pl.vector_function
def floor_vf(dst_tile, val):
    """dst_tile lane 0 = int32(floor(val)) for val >= 0 (astype truncates toward 0)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    f = vf.full(val, dtype=pl.DT_FP32)
    i = vf.astype(f, preg, dtype=pl.DT_INT32)
    vf.store_align(dst_tile, i, preg, 0)


@pl.vector_function
def init_acc_row_vf(acc_tile, pw, init_val):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg = vf.full(init_val, dtype=pl.DT_FP32)
    vf.store_align(acc_tile, vreg, preg, [pw, 0])


@pl.vector_function
def accumulate_avg_vf(acc_tile, t0_tile, t1_tile, pw, xxc, wl1, wy0, wy1):
    """One bilinear sample point; index and weights computed in-register.

    xxc: clamped (>=0) sample x as float; wl1: w_l - 1 (int); wy0/wy1: row weights.
    acc[pw] += wy0*((1-lx)*g00 + lx*g01) + wy1*((1-lx)*g10 + lx*g11).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    f = vf.full(xxc, dtype=pl.DT_FP32)
    x0i = vf.astype(f, preg, dtype=pl.DT_INT32)
    x0f = vf.astype(x0i, preg, dtype=pl.DT_FP32)
    lamx = vf.sub(f, x0f, preg)
    one = vf.full(1.0, dtype=pl.DT_FP32)
    onem = vf.sub(one, lamx, preg)
    onei = vf.full(1, dtype=pl.DT_INT32)
    x1a = vf.add(x0i, onei, preg)
    x1c = vf.mins(x1a, wl1, preg)
    cvec = vf.arange(0, dtype=pl.DT_INT32)
    cvec = vf.muls(cvec, SPAN, preg)
    idx0 = vf.add(cvec, x0i, preg)
    idx1 = vf.add(cvec, x1c, preg)
    g00 = vf.gather(t0_tile, idx0, preg)
    g01 = vf.gather(t0_tile, idx1, preg)
    g10 = vf.gather(t1_tile, idx0, preg)
    g11 = vf.gather(t1_tile, idx1, preg)
    w00 = vf.muls(onem, wy0, preg)
    w01 = vf.muls(lamx, wy0, preg)
    w10 = vf.muls(onem, wy1, preg)
    w11 = vf.muls(lamx, wy1, preg)
    m00 = vf.mul(g00, w00, preg)
    m01 = vf.mul(g01, w01, preg)
    m10 = vf.mul(g10, w10, preg)
    m11 = vf.mul(g11, w11, preg)
    vhi = vf.add(m00, m01, preg)
    vlo = vf.add(m10, m11, preg)
    val = vf.add(vhi, vlo, preg)
    cur = vf.load_align(acc_tile, [pw, 0])
    new = vf.add(cur, val, preg)
    vf.store_align(acc_tile, new, preg, [pw, 0])


@pl.vector_function
def accumulate_max_vf(acc_tile, t0_tile, t1_tile, pw, xxc, wl1, wy0, wy1):
    """acc[pw] = max(acc[pw], bilinear(xxc)) — same in-register index/weights."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    f = vf.full(xxc, dtype=pl.DT_FP32)
    x0i = vf.astype(f, preg, dtype=pl.DT_INT32)
    x0f = vf.astype(x0i, preg, dtype=pl.DT_FP32)
    lamx = vf.sub(f, x0f, preg)
    one = vf.full(1.0, dtype=pl.DT_FP32)
    onem = vf.sub(one, lamx, preg)
    onei = vf.full(1, dtype=pl.DT_INT32)
    x1a = vf.add(x0i, onei, preg)
    x1c = vf.mins(x1a, wl1, preg)
    cvec = vf.arange(0, dtype=pl.DT_INT32)
    cvec = vf.muls(cvec, SPAN, preg)
    idx0 = vf.add(cvec, x0i, preg)
    idx1 = vf.add(cvec, x1c, preg)
    g00 = vf.gather(t0_tile, idx0, preg)
    g01 = vf.gather(t0_tile, idx1, preg)
    g10 = vf.gather(t1_tile, idx0, preg)
    g11 = vf.gather(t1_tile, idx1, preg)
    w00 = vf.muls(onem, wy0, preg)
    w01 = vf.muls(lamx, wy0, preg)
    w10 = vf.muls(onem, wy1, preg)
    w11 = vf.muls(lamx, wy1, preg)
    m00 = vf.mul(g00, w00, preg)
    m01 = vf.mul(g01, w01, preg)
    m10 = vf.mul(g10, w10, preg)
    m11 = vf.mul(g11, w11, preg)
    vhi = vf.add(m00, m01, preg)
    vlo = vf.add(m10, m11, preg)
    val = vf.add(vhi, vlo, preg)
    cur = vf.load_align(acc_tile, [pw, 0])
    new = vf.max(cur, val, preg)
    vf.store_align(acc_tile, new, preg, [pw, 0])


@pl.vector_function
def scale_acc_row_vf(acc_tile, out_tile, pw, inv):
    """avg: out = acc[pw] * inv."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    row = vf.load_align(acc_tile, [pw, 0])
    scaled = vf.muls(row, inv, preg)
    vf.store_align(out_tile, scaled, preg, 0)


@pl.vector_function
def copy_acc_row_vf(acc_tile, out_tile, pw):
    """max: out = acc[pw]."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    row = vf.load_align(acc_tile, [pw, 0])
    vf.store_align(out_tile, row, preg, 0)


# ================================================================
#  Per-level RoI worker — one trace instantiation per level pointer
# ================================================================


def process_level_roi(
    feat_view,
    h_l,
    w_l,
    s_l,
    lvl_param,
    k,
    c0,
    cb_valid,
    rois_t,
    index_t,
    out_view,
    tiling,
    tiles,
):
    """Compute the RoI's mapped level from its coordinates; when it equals
    lvl_param (whose view/pointer is compile-time bound here), run the full
    RoIAlign extraction for channel block [c0, c0+cb_valid)."""
    row_len = 5 - tiling.hasIndex

    if tiling.hasIndex == 1:
        bi_f = pl.getval(index_t, k) * 1.0
        x1 = pl.getval(rois_t, k * row_len + 0) * 1.0
        y1 = pl.getval(rois_t, k * row_len + 1) * 1.0
        x2 = pl.getval(rois_t, k * row_len + 2) * 1.0
        y2 = pl.getval(rois_t, k * row_len + 3) * 1.0
    else:
        bi_f = pl.getval(rois_t, k * row_len + 0) * 1.0
        x1 = pl.getval(rois_t, k * row_len + 1) * 1.0
        y1 = pl.getval(rois_t, k * row_len + 2) * 1.0
        x2 = pl.getval(rois_t, k * row_len + 3) * 1.0
        y2 = pl.getval(rois_t, k * row_len + 4) * 1.0
    # batch index: clamp then float->int via the floor roundtrip (offsets need int)
    bi_hi = (tiling.batchSize - 1) * 1.0
    bi_t = bi_f if bi_f > 0.0 else 0.0
    bi_c = bi_t if bi_t < bi_hi else bi_hi

    # level mapping, comparison form of floor(log2(sqrt(area)/eff + 1e-6))
    rf = tiling.roiScaleFactor
    eff = tiling.finestScale * 1.0 * (rf if rf > 0.0 else 1.0)
    eff2 = eff * eff
    area = (x2 - x1) * (y2 - y1)
    lvl = 0
    if 1 < tiling.numLevels and area >= eff2 * 4.0:
        lvl = 1
    if 2 < tiling.numLevels and area >= eff2 * 16.0:
        lvl = 2
    if 3 < tiling.numLevels and area >= eff2 * 64.0:
        lvl = 3
    if 4 < tiling.numLevels and area >= eff2 * 256.0:
        lvl = 4

    if lvl == lvl_param:
        acc_tile = tiles[0]
        t0f_tile = tiles[1]
        t1f_tile = tiles[2]
        outf_tile = tiles[3]
        outx_tile = tiles[4]
        t0x_tile = tiles[5]
        t1x_tile = tiles[6]
        ints_tile = tiles[7]

        floor_vf(ints_tile, bi_c)
        bi = pl.getval(ints_tile, 0)

        ph_num = tiling.pooledHeight
        pw_num = tiling.pooledWidth
        wl1 = w_l - 1

        off = 0.5 if tiling.aligned == 1 else 0.0
        rx1 = x1 * s_l - off
        ry1 = y1 * s_l - off
        rx2 = x2 * s_l - off
        ry2 = y2 * s_l - off
        rw_a = rx2 - rx1
        rw_b = rw_a if rw_a > 1.0 else 1.0
        rh_a = ry2 - ry1
        rh_b = rh_a if rh_a > 1.0 else 1.0
        bw = rw_b / (pw_num * 1.0)
        bh = rh_b / (ph_num * 1.0)

        # adaptive sample counts: ceil via floor + fractional check
        sn = tiling.sampleNum
        if sn <= 0:
            floor_vf(ints_tile, bh)
            bh_i = pl.getval(ints_tile, 0)
            bh_fr = bh - bh_i * 1.0
            bh_c = bh_i + 1 if bh_fr > 0.0 else bh_i
            nh = bh_c if bh_c > 1 else 1
            floor_vf(ints_tile, bw)
            bw_i = pl.getval(ints_tile, 0)
            bw_fr = bw - bw_i * 1.0
            bw_c = bw_i + 1 if bw_fr > 0.0 else bw_i
            nw = bw_c if bw_c > 1 else 1
        else:
            nh = sn
            nw = sn
        inv_cnt = 1.0 / (nh * nw)

        for ph in pl.range(0, ph_num):
            for pw_i in pl.range(0, pw_num):
                if pool_mode == 0:
                    init_acc_row_vf(acc_tile, pw_i, 0.0)
                else:
                    init_acc_row_vf(acc_tile, pw_i, NEG_INF)
            for ih in pl.range(0, nh):
                yy = ry1 + ph * bh + (ih + 0.5) * bh / nh
                inside_y = yy >= -1.0 and yy <= h_l * 1.0
                yyc = yy if yy > 0.0 else 0.0
                floor_vf(ints_tile, yyc)
                y0 = pl.getval(ints_tile, 0)
                y1a = y0 + 1
                y1r = y1a if y1a < h_l else h_l - 1
                lam_y = yyc - y0 * 1.0
                wy0 = 1.0 - lam_y
                wy1 = lam_y

                # window [0, x_hi]: main path when it fits SPAN columns
                x_last = rx1 + (pw_num - 1) * bw + (nw - 0.5) * bw / nw
                floor_vf(ints_tile, x_last + 2.0)
                x_hi = pl.getval(ints_tile, 0)
                x_hi_c = x_hi if x_hi < wl1 else wl1
                span_need = x_hi_c + 1

                if span_need <= SPAN:
                    # main path: row-pair window load + UB gather
                    if input_dtype == pl.DT_FP32:
                        pl.set_validshape(t0f_tile, [cb_valid, span_need])
                        pl.load(t0f_tile, feat_view, [bi, c0, y0, 0], order=[1, 3])
                        pl.set_validshape(t1f_tile, [cb_valid, span_need])
                        pl.load(t1f_tile, feat_view, [bi, c0, y1r, 0], order=[1, 3])
                    else:
                        pl.set_validshape(t0x_tile, [cb_valid, span_need])
                        pl.load(t0x_tile, feat_view, [bi, c0, y0, 0], order=[1, 3])
                        pl.cast(t0f_tile, t0x_tile)
                        pl.set_validshape(t1x_tile, [cb_valid, span_need])
                        pl.load(t1x_tile, feat_view, [bi, c0, y1r, 0], order=[1, 3])
                        pl.cast(t1f_tile, t1x_tile)
                    for pw_i in pl.range(0, pw_num):
                        for iw in pl.range(0, nw):
                            xx = rx1 + pw_i * bw + (iw + 0.5) * bw / nw
                            inside_x = xx >= -1.0 and xx <= w_l * 1.0
                            if inside_y and inside_x:
                                xxc = xx if xx > 0.0 else 0.0
                                if pool_mode == 0:
                                    accumulate_avg_vf(
                                        acc_tile, t0f_tile, t1f_tile, pw_i,
                                        xxc, wl1, wy0, wy1,
                                    )
                                else:
                                    accumulate_max_vf(
                                        acc_tile, t0f_tile, t1f_tile, pw_i,
                                        xxc, wl1, wy0, wy1,
                                    )
                else:
                    # fallback: per-sample-point 2-column row-pair windows; the same
                    # accumulate_*_vf runs with the fractional coordinate relative to
                    # x0 and wl1_rel clamping the second gather column.
                    for pw_i in pl.range(0, pw_num):
                        for iw in pl.range(0, nw):
                            xx = rx1 + pw_i * bw + (iw + 0.5) * bw / nw
                            inside_x = xx >= -1.0 and xx <= w_l * 1.0
                            if inside_y and inside_x:
                                xxc = xx if xx > 0.0 else 0.0
                                floor_vf(ints_tile, xxc)
                                x0 = pl.getval(ints_tile, 0)
                                xxc_rel = xxc - x0 * 1.0
                                w2 = 2 if x0 < wl1 else 1
                                wl1_rel = 1 if x0 < wl1 else 0
                                if input_dtype == pl.DT_FP32:
                                    pl.set_validshape(t0f_tile, [cb_valid, w2])
                                    pl.load(t0f_tile, feat_view, [bi, c0, y0, x0], order=[1, 3])
                                    pl.set_validshape(t1f_tile, [cb_valid, w2])
                                    pl.load(t1f_tile, feat_view, [bi, c0, y1r, x0], order=[1, 3])
                                else:
                                    pl.set_validshape(t0x_tile, [cb_valid, w2])
                                    pl.load(t0x_tile, feat_view, [bi, c0, y0, x0], order=[1, 3])
                                    pl.cast(t0f_tile, t0x_tile)
                                    pl.set_validshape(t1x_tile, [cb_valid, w2])
                                    pl.load(t1x_tile, feat_view, [bi, c0, y1r, x0], order=[1, 3])
                                    pl.cast(t1f_tile, t1x_tile)
                                if pool_mode == 0:
                                    accumulate_avg_vf(
                                        acc_tile, t0f_tile, t1f_tile, pw_i,
                                        xxc_rel, wl1_rel, wy0, wy1,
                                    )
                                else:
                                    accumulate_max_vf(
                                        acc_tile, t0f_tile, t1f_tile, pw_i,
                                        xxc_rel, wl1_rel, wy0, wy1,
                                    )

            # write output row ph
            for pw_i in pl.range(0, pw_num):
                if pool_mode == 0:
                    scale_acc_row_vf(acc_tile, outf_tile, pw_i, inv_cnt)
                else:
                    copy_acc_row_vf(acc_tile, outf_tile, pw_i)
                pl.set_validshape(outx_tile, [1, cb_valid])
                pl.cast(outx_tile, outf_tile, mode=pl.RoundMode.CAST_RINT)
                pl.store(out_view, outx_tile, [k, ph, pw_i, c0])


# ================================================================
#  Kernel entry — dynamic rank: raw pointers, shapes from tiling
# ================================================================


@pl.jit(
    auto_mutex=True,
    tiling_key=RoiExtractorTilingKey,
    datatype={
        "feats0": "input_dtype",
        "feats1": "input_dtype",
        "feats2": "input_dtype",
        "feats3": "input_dtype",
        "feats4": "input_dtype",
        "rois": "input_dtype",
    },
)
def roi_extractor(
    feats0: pl.Ptr[pl.DT_UINT8],
    feats1: pl.Ptr[pl.DT_UINT8],
    feats2: pl.Ptr[pl.DT_UINT8],
    feats3: pl.Ptr[pl.DT_UINT8],
    feats4: pl.Ptr[pl.DT_UINT8],
    rois: pl.Ptr[pl.DT_UINT8],
    index: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8],
    tiling: RoiExtractorTiling,
):
    with pl.section_vector():
        n_lvl = tiling.numLevels
        k_num = tiling.roisNum
        c_num = tiling.channels
        row_len = 5 - tiling.hasIndex

        rois_t = pl.make_tensor(rois, [k_num * row_len], [1], dtype=input_dtype)
        index_t = pl.make_tensor(index, [k_num], [1], dtype=pl.DT_INT32)
        # permuted view [K, PH, PW, C]: column stores take the default ascending
        # order with a [1, CB] tile (dim1 spans C at stride PH*PW)
        out_view = pl.make_tensor(
            y,
            [k_num, tiling.pooledHeight, tiling.pooledWidth, c_num],
            [c_num * tiling.pooledHeight * tiling.pooledWidth,
             tiling.pooledWidth, 1, tiling.pooledHeight * tiling.pooledWidth],
            dtype=input_dtype,
        )

        # ---- UB tiles ----
        acc_grp = pl.make_tile_group(
            type=pl.TileType(shape=[PW_MAX, CB], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=A_ACC,
            mutex_ids=[0],
        )
        t0f_grp = pl.make_tile_group(
            type=pl.TileType(shape=[CB, SPAN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=A_T0F,
            mutex_ids=[1],
        )
        t1f_grp = pl.make_tile_group(
            type=pl.TileType(shape=[CB, SPAN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=A_T1F,
            mutex_ids=[2],
        )
        outf_grp = pl.make_tile_group(
            type=pl.TileType(shape=[1, CB], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=A_OUTF,
            mutex_ids=[3],
        )
        outx_grp = pl.make_tile_group(
            type=pl.TileType(shape=[1, CB], dtype=input_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=A_OUTX,
            mutex_ids=[9],
        )
        ints_grp = pl.make_tile_group(
            type=pl.TileType(shape=[1, CB], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addrs=A_INTS,
            mutex_ids=[10],
        )
        if input_dtype != pl.DT_FP32:
            t0x_grp = pl.make_tile_group(
                type=pl.TileType(shape=[CB, SPAN], dtype=input_dtype, target_memory=pl.MemorySpace.Vec),
                addrs=A_T0X,
                mutex_ids=[11],
            )
            t1x_grp = pl.make_tile_group(
                type=pl.TileType(shape=[CB, SPAN], dtype=input_dtype, target_memory=pl.MemorySpace.Vec),
                addrs=A_T1X,
                mutex_ids=[12],
            )
        else:
            t0x_grp = t0f_grp
            t1x_grp = t1f_grp

        acc_tile = acc_grp.next()
        t0f_tile = t0f_grp.next()
        t1f_tile = t1f_grp.next()
        outf_tile = outf_grp.next()
        outx_tile = outx_grp.next()
        ints_tile = ints_grp.next()
        t0x_tile = t0x_grp.next()
        t1x_tile = t1x_grp.next()
        tiles = (acc_tile, t0f_tile, t1f_tile, outf_tile, outx_tile, t0x_tile, t1x_tile, ints_tile)

        # ---- per-level NCHW views (shapes from tiling fields) ----
        view0 = pl.make_tensor(
            feats0,
            [tiling.batchSize, c_num, tiling.levelH0, tiling.levelW0],
            [c_num * tiling.levelH0 * tiling.levelW0, tiling.levelH0 * tiling.levelW0,
             tiling.levelW0, 1],
            dtype=input_dtype,
        )
        view1 = pl.make_tensor(
            feats1,
            [tiling.batchSize, c_num, tiling.levelH1, tiling.levelW1],
            [c_num * tiling.levelH1 * tiling.levelW1, tiling.levelH1 * tiling.levelW1,
             tiling.levelW1, 1],
            dtype=input_dtype,
        )
        view2 = pl.make_tensor(
            feats2,
            [tiling.batchSize, c_num, tiling.levelH2, tiling.levelW2],
            [c_num * tiling.levelH2 * tiling.levelW2, tiling.levelH2 * tiling.levelW2,
             tiling.levelW2, 1],
            dtype=input_dtype,
        )
        view3 = pl.make_tensor(
            feats3,
            [tiling.batchSize, c_num, tiling.levelH3, tiling.levelW3],
            [c_num * tiling.levelH3 * tiling.levelW3, tiling.levelH3 * tiling.levelW3,
             tiling.levelW3, 1],
            dtype=input_dtype,
        )
        view4 = pl.make_tensor(
            feats4,
            [tiling.batchSize, c_num, tiling.levelH4, tiling.levelW4],
            [c_num * tiling.levelH4 * tiling.levelW4, tiling.levelH4 * tiling.levelW4,
             tiling.levelW4, 1],
            dtype=input_dtype,
        )

        # ---- work distribution: (k, channel block) over cores ----
        core_id = pl.get_block_idx() // pl.get_subblock_num()
        block_num = pl.get_block_num() // pl.get_subblock_num()
        num_cblocks = (c_num + CB - 1) // CB
        total_work = k_num * num_cblocks
        base_work = total_work // block_num
        extra_work = total_work % block_num
        if core_id < extra_work:
            work_start = core_id * (base_work + 1)
            work_count = base_work + 1
        else:
            work_start = extra_work * (base_work + 1) + (core_id - extra_work) * base_work
            work_count = base_work

        for w in pl.range(0, work_count):
            work_id = work_start + w
            k = work_id // num_cblocks
            cb = work_id % num_cblocks
            c0 = cb * CB
            cb_valid = pl.min(c_num - c0, CB)

            if n_lvl > 0:
                process_level_roi(
                    view0, tiling.levelH0, tiling.levelW0, tiling.spatialScale0,
                    0, k, c0, cb_valid, rois_t, index_t, out_view, tiling, tiles,
                )
            if n_lvl > 1:
                process_level_roi(
                    view1, tiling.levelH1, tiling.levelW1, tiling.spatialScale1,
                    1, k, c0, cb_valid, rois_t, index_t, out_view, tiling, tiles,
                )
            if n_lvl > 2:
                process_level_roi(
                    view2, tiling.levelH2, tiling.levelW2, tiling.spatialScale2,
                    2, k, c0, cb_valid, rois_t, index_t, out_view, tiling, tiles,
                )
            if n_lvl > 3:
                process_level_roi(
                    view3, tiling.levelH3, tiling.levelW3, tiling.spatialScale3,
                    3, k, c0, cb_valid, rois_t, index_t, out_view, tiling, tiles,
                )
            if n_lvl > 4:
                process_level_roi(
                    view4, tiling.levelH4, tiling.levelW4, tiling.spatialScale4,
                    4, k, c0, cb_valid, rois_t, index_t, out_view, tiling, tiles,
                )
        return
