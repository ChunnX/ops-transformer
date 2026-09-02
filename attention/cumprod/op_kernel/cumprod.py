# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Cumprod kernel (pypto-pro, single launch).

Inclusive cumulative product along an axis:
    y[..., k, ...] = prod(x[..., 0..k, ...], axis=dim)

Layout normalization: [outer, scan, inner]
    outer = prod(shape[:dim]); scan = shape[dim]; inner = prod(shape[dim+1:])

Axis handling (official Cumprod IR: axis is an INPUT, ValueDepend(REQUIRED)):
    axis_known == 1 (aclnn flow: the api scalar is materialized into a host
        const tensor via executor ConvertToTensor; GE flow: constant-folded
        Const): the host tiling reads it, folds the shape to [outer, scan,
        inner] and computes the segmentation; scan_mode picks the FOLD/SCAN
        body at compile time.
    axis_known == 0 (device-resident axis tensor): the kernel reads it from
        GM, folds the three axes and computes the segmentation at runtime
        (scan_mode bit unused, both bodies present).

Kernel bodies (same launch for every input shape):
    FOLD (inner > 1): lane = inner; per scan-row acc = acc * x_row fold,
        no cross-lane movement.
    SCAN (inner == 1): lane = scan line; per column gather -> (widen) -> mul
        -> (narrow) -> scatter. Single segment: carry = 1. Segmented
        (seg_count > 1): inside the same launch, pass A aggregates each
        segment per scan line into GM workspace [seg_count, outer] (fp32),
        a cross-core stage barrier (set_cross_core(MTE3) -> wait_cross_core
        (MTE2), INTER_BLOCK, event 0, outside every job loop, + dcci) runs
        between the passes, then pass B re-reads x and folds each segment
        with the exclusive prefix product of preceding segments as carry.

Numerical policy (AP-007): fp16/bf16 widen to fp32 on load, chain-multiply
in fp32 and narrow back per element with a single round-to-nearest-even
(segmented pass B re-reads x, never cascading pass A results); fp32 native.
Cross-core carry (workspace) stays fp32.

BlockDim contract: the host always launches all AIV cores (the INTER_BLOCK
barrier requires full-core participation); job striding uses the flattened
AIV index get_block_idx() and worker count get_block_num() *
get_subblock_num(), valid for both AIV-only and MIX launches.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType

# The offline codegen loader (pypto_codegen.py) execs this file without
# registering it in sys.modules; postponed annotations (required by the
# int[N] TilingData array field) then break dataclass processing. Register a
# stand-in module entry before defining the dataclass.
if __name__ not in sys.modules:
    _self_module = ModuleType(__name__)
    _self_module.__dict__.update(globals())
    sys.modules[__name__] = _self_module

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


# ================================================================
#  Tile dimensions and constants
# ================================================================
ROWS32 = 64        # fp32: SCAN scan lines per job (= register lanes)
ROWS16 = 128       # fp16/bf16: SCAN scan lines per job
W32 = 64           # fp32: SCAN column chunk width (1 register)
W16 = 128          # narrow: SCAN column chunk width (1 register)
WP32 = 72          # fp32 gather/scatter tile physical row pitch: 72*4B=288B=9*32B (odd multiple, bank-conflict safe)
WP16 = 144         # narrow gather/scatter tile physical row pitch: 144*2B=288B=9*32B
WS_P32 = 72        # ws prefix tile row pitch (fp32 class, fits ROWS32=64)
WS_P16 = 136       # ws prefix tile row pitch (narrow class, fits ROWS16=128): 136*4B=544B=17*32B
MAXP = 64          # max segments per scan line (ws tile row limit)
TILE_S = 8         # FOLD: scan rows per tile
CHUNK32 = 64       # FOLD: inner chunk (fp32, 1 register)
CHUNK16 = 128      # FOLD: inner chunk (narrow, 1 register)
EVT_AGGREADY = 0   # pass A -> pass B cross-core stage barrier event id (only event in the kernel)

# On-chip address map (Vec space; 32B-aligned, worst case narrow class fits 248KB UB)
A32_SCAN_IN0, A32_SCAN_IN1 = 0x00000, 0x04800
A32_SCAN_OUT0, A32_SCAN_OUT1 = 0x09000, 0x0D800
A32_WS = 0x12000
A32_CARRY = 0x16800
A32_FOLD_IN0, A32_FOLD_IN1 = 0x16900, 0x17100
A32_FOLD_OUT0, A32_FOLD_OUT1 = 0x17900, 0x18100
A16_SCAN_IN0, A16_SCAN_IN1 = 0x00000, 0x09000
A16_SCAN_OUT0, A16_SCAN_OUT1 = 0x12000, 0x1B000
A16_WS = 0x24000
A16_CARRY = 0x2C800
A16_FOLD_IN0, A16_FOLD_IN1 = 0x2CA00, 0x2D200
A16_FOLD_OUT0, A16_FOLD_OUT1 = 0x2DA00, 0x2E200


# ================================================================
#  Tiling data (host fills; layout must match op_host TILING_DATA_DEF)
# ================================================================
@dataclass
class CumprodTiling:
    outer: int       # prod(shape[:axis])     (valid when axis_known)
    scan: int        # shape[axis]            (valid when axis_known)
    inner: int       # prod(shape[axis+1:])   (valid when axis_known)
    seg_count: int   # segments per scan line (valid when axis_known)
    seg_len: int     # segment length         (valid when axis_known)
    rank: int        # rank of x (always valid; device path folds dims itself)
    dims: int[8]


# ================================================================
#  TilingKey
# ================================================================
class CumprodTilingKey:
    axis_known = TilingKeyField(bits=1, values=[0, 1])   # bit0: 1 = host-resolved axes
    scan_mode = TilingKeyField(bits=1, values=[0, 1])    # bit1: 0 = FOLD, 1 = SCAN (used when axis_known=1)


# ================================================================
#  VF helpers (io_dtype is the kernel datatype variable, a compile-time
#  constant; dtype branches are pruned at trace time)
# ================================================================
@pl.vector_function
def fold_rows_vf(in_tile, out_tile, carry_tile,
                 valid_rows: pl.DT_INT64, valid_cols: pl.DT_INT64,
                 init_mode: pl.DT_INT64):
    """FOLD (inner>1): lane = inner, per scan-row acc = acc * x_row fold.

    init_mode: 0 = seed 1.0 (multiplicative identity); 1 = resume from carry tile.
    """
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    if io_dtype == pl.DT_FP32:
        if init_mode == 0:
            acc = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        else:
            acc = vf.load_align(carry_tile, 0)
        for r in pl.range(0, valid_rows):
            x_reg = vf.load_align(in_tile, r * CHUNK32)
            acc = vf.mul(acc, x_reg, preg32)
            vf.store_align(out_tile, acc, preg32, [r, 0])
        vf.store_align(carry_tile, acc, preg32)
    else:
        preg_n = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)
        if init_mode == 0:
            acc_lo = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            acc_hi = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        else:
            acc_lo = vf.load_align(carry_tile, 0)
            acc_hi = vf.load_align(carry_tile, 64)
        for r in pl.range(0, valid_rows):
            x_n = vf.load_align(in_tile, r * CHUNK16)
            lo = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
            hi = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
            acc_lo = vf.mul(acc_lo, lo, preg32)
            acc_hi = vf.mul(acc_hi, hi, preg32)
            out_lo = vf.astype(acc_lo, preg32, dtype=io_dtype, layout=pl.CastLayout.ZERO,
                               round_mode=pl.VFRoundMode.CAST_RINT,
                               saturate=pl.SaturateMode.OFF)
            out_hi = vf.astype(acc_hi, preg32, dtype=io_dtype, layout=pl.CastLayout.ONE,
                               round_mode=pl.VFRoundMode.CAST_RINT,
                               saturate=pl.SaturateMode.OFF)
            out_n = vf.xor(out_lo, out_hi, preg_n)   # zero-complementary merge
            vf.store_align(out_tile, out_n, preg_n, [r, 0])
        vf.store_align(carry_tile, acc_lo, preg32)
        vf.store_align(carry_tile, acc_hi, preg32, [0, 64])
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def scan_cols_vf(in_tile, out_tile, carry_tile, ws_tile,
                 valid_rows: pl.DT_INT64, valid_cols: pl.DT_INT64,
                 init_mode: pl.DT_INT64, seg_idx: pl.DT_INT64):
    """SCAN (inner==1): lane = scan line, per column gather -> (widen) -> mul -> (narrow) -> scatter.

    init_mode: 0 = seed 1.0; 1 = resume from carry tile; 2 = gather-multiply the
    exclusive prefix product of preceding segment aggregates from ws_tile.
    Row-width masks bound the gather/scatter lanes to valid_rows.
    """
    if io_dtype == pl.DT_FP32:
        preg32 = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
        preg_u = vf.update_mask(valid_rows, dtype=pl.DT_UINT32)
        g = vf.arange(0, dtype=pl.DT_UINT32)
        g = vf.muls(g, WP32, preg_u)                 # lane L -> row L start (elements)
        if init_mode == 0:
            acc = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        elif init_mode == 1:
            acc = vf.load_align(carry_tile, 0)
        else:
            g_ws = vf.arange(0, dtype=pl.DT_UINT32)
            acc = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            for k in pl.range(0, seg_idx):
                idx = vf.adds(g_ws, k * WS_P32, preg_u)
                c = vf.gather(ws_tile, idx, preg32)
                acc = vf.mul(acc, c, preg32)
        for j in pl.range(0, valid_cols):
            x_r = vf.gather(in_tile, g, preg32)
            acc = vf.mul(acc, x_r, preg32)
            vf.scatter(out_tile, acc, g, preg32)
            g = vf.adds(g, 1, preg_u)
        vf.store_align(carry_tile, acc, preg32)
    else:
        preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        preg_u32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
        preg_n = vf.update_mask(valid_rows, dtype=io_dtype)
        preg_u = vf.update_mask(valid_rows, dtype=pl.DT_UINT16)
        g = vf.arange(0, dtype=pl.DT_UINT16)
        g = vf.muls(g, WP16, preg_u)                 # lane L -> row L start (elements)
        if init_mode == 0:
            acc_lo = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            acc_hi = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        elif init_mode == 1:
            acc_lo = vf.load_align(carry_tile, 0)
            acc_hi = vf.load_align(carry_tile, 64)
        else:
            # ws is fp32: even/odd scan lines gather twice (lane j <-> rows 2j / 2j+1)
            g_ws = vf.arange(0, dtype=pl.DT_UINT32)
            g_ws = vf.muls(g_ws, 2, preg_u32)
            acc_lo = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            acc_hi = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            for k in pl.range(0, seg_idx):
                idx_lo = vf.adds(g_ws, k * WS_P16, preg_u32)
                c_lo = vf.gather(ws_tile, idx_lo, preg32)
                acc_lo = vf.mul(acc_lo, c_lo, preg32)
                idx_hi = vf.adds(g_ws, k * WS_P16 + 1, preg_u32)
                c_hi = vf.gather(ws_tile, idx_hi, preg32)
                acc_hi = vf.mul(acc_hi, c_hi, preg32)
        for j in pl.range(0, valid_cols):
            x_n = vf.gather(in_tile, g, preg_n)
            lo = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
            hi = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
            acc_lo = vf.mul(acc_lo, lo, preg32)
            acc_hi = vf.mul(acc_hi, hi, preg32)
            out_lo = vf.astype(acc_lo, preg32, dtype=io_dtype, layout=pl.CastLayout.ZERO,
                               round_mode=pl.VFRoundMode.CAST_RINT,
                               saturate=pl.SaturateMode.OFF)
            out_hi = vf.astype(acc_hi, preg32, dtype=io_dtype, layout=pl.CastLayout.ONE,
                               round_mode=pl.VFRoundMode.CAST_RINT,
                               saturate=pl.SaturateMode.OFF)
            out_n = vf.xor(out_lo, out_hi, preg_n)   # zero-complementary merge
            vf.scatter(out_tile, out_n, g, preg_n)
            g = vf.adds(g, 1, preg_u)
        vf.store_align(carry_tile, acc_lo, preg32)
        vf.store_align(carry_tile, acc_hi, preg32, [0, 64])
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def agg_cols_vf(in_tile, carry_tile,
                valid_rows: pl.DT_INT64, valid_cols: pl.DT_INT64,
                init_mode: pl.DT_INT64, is_final: pl.DT_INT64):
    """Pass A segment aggregation (inner==1 and seg_count>1, in-kernel SCAN prologue).

    init_mode: 0 = seed 1.0; 1 = resume from carry tile. is_final=1 (job's last
    chunk): narrow class stores acc_lo/acc_hi interleaved into the carry tile in
    scan-line order (INTLV_B32) so pl.store writes GM ws[s, r0+L] directly;
    fp32 class has a single acc already in row order.
    Aggregation stays fp32 (cross-core carry is never narrowed).
    """
    if io_dtype == pl.DT_FP32:
        preg32 = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
        preg_u = vf.update_mask(valid_rows, dtype=pl.DT_UINT32)
        g = vf.arange(0, dtype=pl.DT_UINT32)
        g = vf.muls(g, WP32, preg_u)
        if init_mode == 0:
            acc = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        else:
            acc = vf.load_align(carry_tile, 0)
        for j in pl.range(0, valid_cols):
            x_r = vf.gather(in_tile, g, preg32)
            acc = vf.mul(acc, x_r, preg32)
            g = vf.adds(g, 1, preg_u)
        vf.store_align(carry_tile, acc, preg32)      # lane L = row L
    else:
        preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        preg_n = vf.update_mask(valid_rows, dtype=io_dtype)
        preg_u = vf.update_mask(valid_rows, dtype=pl.DT_UINT16)
        g = vf.arange(0, dtype=pl.DT_UINT16)
        g = vf.muls(g, WP16, preg_u)
        if init_mode == 0:
            acc_lo = vf.full(1.0, preg32, dtype=pl.DT_FP32)
            acc_hi = vf.full(1.0, preg32, dtype=pl.DT_FP32)
        else:
            acc_lo = vf.load_align(carry_tile, 0)
            acc_hi = vf.load_align(carry_tile, 64)
        for j in pl.range(0, valid_cols):
            x_n = vf.gather(in_tile, g, preg_n)
            lo = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
            hi = vf.astype(x_n, preg_n, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
            acc_lo = vf.mul(acc_lo, lo, preg32)
            acc_hi = vf.mul(acc_hi, hi, preg32)
            g = vf.adds(g, 1, preg_u)
        if is_final == 1:
            # interleaved dual store: stage[2j] = acc_lo[j] (row 2j), stage[2j+1] = acc_hi[j] (row 2j+1)
            vf.store_align(carry_tile, acc_lo, acc_hi, preg32,
                           dist=pl.StoreDist.INTLV_B32)
        else:
            vf.store_align(carry_tile, acc_lo, preg32)
            vf.store_align(carry_tile, acc_hi, preg32, [0, 64])
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  Kernel bodies — plain helpers traced (inlined) into the kernel
# ================================================================
def _fold_body(x2, y2, outer, scan, inner, chunk, fold_in, fold_out, carry_slot,
               core_id, num_cores):
    """FOLD (inner>1): jobs = outer * ceil(inner/chunk)."""
    n_chunk = (inner + chunk - 1) // chunk
    total_jobs = outer * n_chunk
    for job in pl.range(core_id, total_jobs, num_cores):
        o = job // n_chunk
        c0 = (job % n_chunk) * chunk
        valid_cols = pl.min(chunk, inner - c0)
        for s0 in pl.range(0, scan, TILE_S):
            valid_rows = pl.min(TILE_S, scan - s0)
            in_slot = fold_in.next()
            pl.set_validshape(in_slot, [valid_rows, valid_cols])
            pl.load(in_slot, x2, [o * scan + s0, c0])
            out_slot = fold_out.next()
            pl.set_validshape(out_slot, [valid_rows, valid_cols])
            init_mode = 1
            if s0 == 0:
                init_mode = 0
            fold_rows_vf(in_slot, out_slot, carry_slot,
                         valid_rows, valid_cols, init_mode)
            pl.store(y2, out_slot, [o * scan + s0, c0])


def _scan_body(x1, y1, ws, outer, scan, seg_count, seg_len, rows, w,
               scan_in, scan_out, ws_group, carry_slot, core_id, num_cores):
    """SCAN (inner==1): jobs = ceil(outer/rows) * seg_count; seg_count>1 runs
    pass A -> cross-core stage barrier -> pass B inside the same launch."""
    row_blocks = (outer + rows - 1) // rows
    total_jobs = row_blocks * seg_count

    if seg_count > 1:
        # ---- pass A: segment aggregation, ws[s, r0:r0+valid_rows] (fp32) ----
        for job in pl.range(core_id, total_jobs, num_cores):
            rb = job // seg_count
            s = job % seg_count
            r0 = rb * rows
            seg_beg = s * seg_len
            seg_end = pl.min(scan, seg_beg + seg_len)
            valid_rows = pl.min(rows, outer - r0)
            for c0 in pl.range(seg_beg, seg_end, w):
                valid_cols = pl.min(w, seg_end - c0)
                in_slot = scan_in.next()      # shared with pass B (same group)
                pl.set_validshape(in_slot, [valid_rows, valid_cols])
                pl.load(in_slot, x1, [r0, c0])
                init_mode = 1
                if c0 == seg_beg:
                    init_mode = 0
                is_final = 0
                if c0 + w >= seg_end:
                    is_final = 1
                agg_cols_vf(in_slot, carry_slot, valid_rows, valid_cols,
                            init_mode, is_final)
            pl.set_validshape(carry_slot, [1, valid_rows])
            pl.store(ws, carry_slot, [s, r0])

        # ---- cross-core stage barrier: outside every runtime loop, executed by
        # every launched core exactly once (seg_count is uniform across cores) ----
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3, event_id=EVT_AGGREADY,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE2, event_id=EVT_AGGREADY,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK)
        pl.system.dcci(ws)                    # clean + invalidate data cache

    # ---- pass B: main path, always runs; single segment uses carry=1, no ws ----
    for job in pl.range(core_id, total_jobs, num_cores):
        rb = job // seg_count
        s = job % seg_count
        r0 = rb * rows
        seg_beg = s * seg_len
        seg_end = pl.min(scan, seg_beg + seg_len)
        valid_rows = pl.min(rows, outer - r0)
        ws_slot = ws_group.current()
        if seg_count > 1:
            pl.set_validshape(ws_slot, [seg_count, valid_rows])
            pl.load(ws_slot, ws, [0, r0])
        for c0 in pl.range(seg_beg, seg_end, w):
            valid_cols = pl.min(w, seg_end - c0)
            in_slot = scan_in.next()
            pl.set_validshape(in_slot, [valid_rows, valid_cols])
            pl.load(in_slot, x1, [r0, c0])
            out_slot = scan_out.next()
            pl.set_validshape(out_slot, [valid_rows, valid_cols])
            init_mode = 1
            if c0 == seg_beg:
                init_mode = 0
                if seg_count > 1:
                    init_mode = 2             # fold segment prefix from ws (re-read x)
            scan_cols_vf(in_slot, out_slot, carry_slot, ws_slot,
                         valid_rows, valid_cols, init_mode, s)
            pl.store(y1, out_slot, [r0, c0])


# ================================================================
#  Kernel — raw pointers, shapes come from tiling / axis tensor
# ================================================================
@pl.jit(
    auto_mutex=True,
    tiling_key=CumprodTilingKey,
    datatype={
        "x": "io_dtype",
        "y": "io_dtype",
    },
)
def cumprod(
    x: pl.Ptr[pl.DT_UINT8],
    axis: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: CumprodTiling,
):
    # io_dtype is a compile-time constant: tile sizes / addresses branch at trace time
    if io_dtype == pl.DT_FP32:
        ROWS = ROWS32
        W = W32
        CHUNK = CHUNK32
        # Tile groups per dtype class: addrs use module-level single base
        # addresses (slots are contiguous: base + i * slot_size).
        io_tt = pl.TileType(shape=[ROWS32, WP32], dtype=io_dtype,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        scan_in = pl.make_tile_group(type=io_tt, addrs=A32_SCAN_IN0, mutex_ids=[0, 1])
        scan_out = pl.make_tile_group(type=io_tt, addrs=A32_SCAN_OUT0, mutex_ids=[2, 3])
        ws_group = pl.make_tile_group(
            type=pl.TileType(shape=[MAXP, WS_P32], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
            addrs=A32_WS, mutex_ids=[4])
        carry_group = pl.make_tile_group(
            type=pl.TileType(shape=[1, ROWS32], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
            addrs=A32_CARRY, mutex_ids=[5])
        fold_tt = pl.TileType(shape=[TILE_S, CHUNK32], dtype=io_dtype,
                              target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        fold_in = pl.make_tile_group(type=fold_tt, addrs=A32_FOLD_IN0, mutex_ids=[6, 7])
        fold_out = pl.make_tile_group(type=fold_tt, addrs=A32_FOLD_OUT0, mutex_ids=[8, 9])
    else:
        ROWS = ROWS16
        W = W16
        CHUNK = CHUNK16
        io_tt = pl.TileType(shape=[ROWS16, WP16], dtype=io_dtype,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        scan_in = pl.make_tile_group(type=io_tt, addrs=A16_SCAN_IN0, mutex_ids=[0, 1])
        scan_out = pl.make_tile_group(type=io_tt, addrs=A16_SCAN_OUT0, mutex_ids=[2, 3])
        ws_group = pl.make_tile_group(
            type=pl.TileType(shape=[MAXP, WS_P16], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
            addrs=A16_WS, mutex_ids=[4])
        carry_group = pl.make_tile_group(
            type=pl.TileType(shape=[1, ROWS16], dtype=pl.DT_FP32,
                             target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
            addrs=A16_CARRY, mutex_ids=[5])
        fold_tt = pl.TileType(shape=[TILE_S, CHUNK16], dtype=io_dtype,
                              target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
        fold_in = pl.make_tile_group(type=fold_tt, addrs=A16_FOLD_IN0, mutex_ids=[6, 7])
        fold_out = pl.make_tile_group(type=fold_tt, addrs=A16_FOLD_OUT0, mutex_ids=[8, 9])

    # ---- three-fold axis resolution ----
    if axis_known == 1:
        # Host tiling read the axis (aclnn materialized host const / GE constant).
        outer = tiling.outer
        scan = tiling.scan
        inner = tiling.inner
    else:
        # axis is device-resident: read it and fold the shape at runtime. The fold is
        # unrolled (rank <= 8) straight-line arithmetic WITHOUT loop-carried phis or
        # name rebinding: make_tensor stride args capture the first binding of a name.
        tensor_axis = pl.make_tensor(axis, [1], [1], dtype=pl.DT_INT64)
        axis_val = pl.getval(tensor_axis, 0)
        if axis_val < 0:
            axis_val = axis_val + tiling.rank
        # per-dim selection masks (pure int arithmetic, no branches):
        #   mlt_i = 1 iff i < axis_val (multiplies into outer)
        #   meq_i = 1 iff i == axis_val (multiplies into scan)
        #   mgt_i = 1 iff i > axis_val (multiplies into inner; guarded by min(d_i,1)
        #           so zero-filled tail dims [rank..7] are never selected)
        mlt0 = pl.min(pl.max(axis_val - 0, 0), 1)
        meq0 = 1 - pl.min((axis_val - 0) * (axis_val - 0), 1)
        mgt0 = pl.min(pl.max(0 - axis_val, 0), 1) * pl.min(tiling.dims[0], 1)
        mlt1 = pl.min(pl.max(axis_val - 1, 0), 1)
        meq1 = 1 - pl.min((axis_val - 1) * (axis_val - 1), 1)
        mgt1 = pl.min(pl.max(1 - axis_val, 0), 1) * pl.min(tiling.dims[1], 1)
        mlt2 = pl.min(pl.max(axis_val - 2, 0), 1)
        meq2 = 1 - pl.min((axis_val - 2) * (axis_val - 2), 1)
        mgt2 = pl.min(pl.max(2 - axis_val, 0), 1) * pl.min(tiling.dims[2], 1)
        mlt3 = pl.min(pl.max(axis_val - 3, 0), 1)
        meq3 = 1 - pl.min((axis_val - 3) * (axis_val - 3), 1)
        mgt3 = pl.min(pl.max(3 - axis_val, 0), 1) * pl.min(tiling.dims[3], 1)
        mlt4 = pl.min(pl.max(axis_val - 4, 0), 1)
        meq4 = 1 - pl.min((axis_val - 4) * (axis_val - 4), 1)
        mgt4 = pl.min(pl.max(4 - axis_val, 0), 1) * pl.min(tiling.dims[4], 1)
        mlt5 = pl.min(pl.max(axis_val - 5, 0), 1)
        meq5 = 1 - pl.min((axis_val - 5) * (axis_val - 5), 1)
        mgt5 = pl.min(pl.max(5 - axis_val, 0), 1) * pl.min(tiling.dims[5], 1)
        mlt6 = pl.min(pl.max(axis_val - 6, 0), 1)
        meq6 = 1 - pl.min((axis_val - 6) * (axis_val - 6), 1)
        mgt6 = pl.min(pl.max(6 - axis_val, 0), 1) * pl.min(tiling.dims[6], 1)
        mlt7 = pl.min(pl.max(axis_val - 7, 0), 1)
        meq7 = 1 - pl.min((axis_val - 7) * (axis_val - 7), 1)
        mgt7 = pl.min(pl.max(7 - axis_val, 0), 1) * pl.min(tiling.dims[7], 1)
        d0 = tiling.dims[0]
        d1 = tiling.dims[1]
        d2 = tiling.dims[2]
        d3 = tiling.dims[3]
        d4 = tiling.dims[4]
        d5 = tiling.dims[5]
        d6 = tiling.dims[6]
        d7 = tiling.dims[7]
        outer = (d0 * mlt0 + (1 - mlt0)) * (d1 * mlt1 + (1 - mlt1)) * (d2 * mlt2 + (1 - mlt2)) * (d3 * mlt3 + (1 - mlt3)) * (d4 * mlt4 + (1 - mlt4)) * (d5 * mlt5 + (1 - mlt5)) * (d6 * mlt6 + (1 - mlt6)) * (d7 * mlt7 + (1 - mlt7))
        scan = (d0 * meq0 + (1 - meq0)) * (d1 * meq1 + (1 - meq1)) * (d2 * meq2 + (1 - meq2)) * (d3 * meq3 + (1 - meq3)) * (d4 * meq4 + (1 - meq4)) * (d5 * meq5 + (1 - meq5)) * (d6 * meq6 + (1 - meq6)) * (d7 * meq7 + (1 - meq7))
        inner = (d0 * mgt0 + (1 - mgt0)) * (d1 * mgt1 + (1 - mgt1)) * (d2 * mgt2 + (1 - mgt2)) * (d3 * mgt3 + (1 - mgt3)) * (d4 * mgt4 + (1 - mgt4)) * (d5 * mgt5 + (1 - mgt5)) * (d6 * mgt6 + (1 - mgt6)) * (d7 * mgt7 + (1 - mgt7))

    # ---- GM views (element strides; host never touches tensor data) ----
    x2 = pl.make_tensor(x, [outer * scan, inner], [inner, 1], dtype=io_dtype)
    y2 = pl.make_tensor(y, [outer * scan, inner], [inner, 1], dtype=io_dtype)
    x1 = pl.make_tensor(x, [outer, scan], [scan, 1], dtype=io_dtype)
    y1 = pl.make_tensor(y, [outer, scan], [scan, 1], dtype=io_dtype)

    with pl.section_vector():
        num_cores = pl.get_block_num() * pl.get_subblock_num()
        core_id = pl.get_block_idx()

        # ---- segmentation ----
        if axis_known == 1:
            seg_count = tiling.seg_count
            seg_len = tiling.seg_len
        else:
            # Device-side segmentation (same formula as host tiling): aim for
            # total jobs ~ 2 * cores; bounded by the workspace the host reserved.
            jobs_base = (outer + ROWS - 1) // ROWS
            jobs_base = pl.max(jobs_base, 1)
            seg_count = pl.min(pl.min(MAXP, scan), (2 * num_cores + jobs_base - 1) // jobs_base)
            seg_count = pl.max(seg_count, 1)
            seg_len = (scan + seg_count - 1) // seg_count
            seg_len = pl.max(seg_len, 1)
            seg_count = (scan + seg_len - 1) // seg_len
        ws = pl.make_tensor(workspace, [seg_count, outer], [outer, 1], dtype=pl.DT_FP32)
        carry_slot = carry_group.current()

        # ---- dispatch ----
        if axis_known == 1:
            if scan_mode == 0:
                _fold_body(x2, y2, outer, scan, inner, CHUNK,
                           fold_in, fold_out, carry_slot, core_id, num_cores)
            else:
                _scan_body(x1, y1, ws, outer, scan, seg_count, seg_len, ROWS, W,
                           scan_in, scan_out, ws_group, carry_slot, core_id, num_cores)
        else:
            if inner == 1:
                _scan_body(x1, y1, ws, outer, scan, seg_count, seg_len, ROWS, W,
                           scan_in, scan_out, ws_group, carry_slot, core_id, num_cores)
            else:
                _fold_body(x2, y2, outer, scan, inner, CHUNK,
                           fold_in, fold_out, carry_slot, core_id, num_cores)
