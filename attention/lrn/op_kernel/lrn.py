# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""LRN kernel (PyPTO-Pro) — AscendC delivery port of the PyPTO-Pro Stage-4
LRN kernel.

Port source (algorithm + tile design, verbatim): ``custom/LRN/test_LRN.py``
of the pypto-pro workflow (Stage 4 verifier PASS 15/15; 16/16 precision
cases vs the CPU golden, atol=rtol=1e-3). This file adapts that kernel to
the ops-transformer binary delivery contract; the math, tile layout, UB
address map and VF chains are unchanged:

  * y = x / (bias + alpha * sum_{j in win(c)} x_j^2)^beta with the clipped
    channel window win(c) = [max(0, c-r), min(C-1, c+r)] — TF/GEIR
    semantics (alpha NOT divided by the window width); the power runs the
    stable exp(-beta * ln(norm)) path in the FP32 domain.
  * fp16 combo: astype(ZERO/ONE) widen -> fp32 square/accumulate ->
    muls/adds -> ln/muls(-beta)/exp -> mul -> INTLV_B32 staging store into
    an fp32 tile -> pl.cast(CAST_RINT). fp32 combo: no astype / no INTLV /
    no cast. Compile-time datatype specialization
    (datatype={"x": "io_dtype", "y": "io_dtype"}); the io_dtype branch is
    folded per dtype instance, so the two chains never coexist in one
    compiled binary.
  * H*W merge INSIDE the kernel via pl.make_tensor ([N*C, HW] view); the
    host does no view/reshape/cast.
  * Channel-window square-sum via whole-row-multiple vf.load_align offsets
    with dynamic loop bounds absorbed into an outer-scalar base
    (sub-register lane shifts are forbidden — fault 507035).
  * Tail handling: physical tiles stay legal ([c_ext,hw_tile] /
    [c_chunk,hw_tile] = [64..160,128] / [64,128] on a5 — c_ext shrinks with
    the RTier halo tier, 32B aligned); runtime
    valid windows via set_validshape before every load / compute / store;
    VF runs full width (per-lane independent, garbage never reaches GM).
  * Multi-core: flattened task grid N x ceil(C/c_chunk) x ceil(HW/hw_tile),
    grid-stride pl.range(core_id, total_tasks, num_cores); the host tiling
    launches blockDim = min(aivNum, total_tasks) (never hard-coded). The
    tile geometry rides the TilingKey (per-instance compile-time constants,
    docs tiling_result_transfer.md): HwTile (fp16 lanes per VF register,
    fixed at the a5 width on both sides — the host literal LRN_HW_TILE
    mirrors this candidate, and the packed-key validation fail-closes any
    one-sided drift), CChunk=64 (design-frozen scheduling choice —
    multi-core splitting first, not UB maximization) and RTier, the halo
    tier selected as the smallest candidate >= the ACTUAL depth_radius,
    so the baked x-slot reservation shrinks from the worst case
    (c_chunk + 2*48 = 160 rows) to c_chunk + 2*RTier. The host checks
    the exact per-(tier, dtype) UB span against the runtime UB —
    same-source same-tier as the baked
    address map, i.e. necessary AND sufficient.

Delivery-contract adaptations (mechanical, no math change):
  * all scalar parameters moved into the LRNTiling tiling struct (n/c/hw/r
    as int64 INDEX fields; bias/alpha/beta as FP32 fields — the original
    kernel scalars were INT64 shape dims, an INT32 depth_radius and FP32
    bias/alpha/beta, so the value domains are unchanged); the tile
    geometry (hw_tile / c_chunk / halo tier) rides the TilingKey instead
    (compile-time per instance; see LRNTilingKey);
  * the GE task arg layout is <tensors> + <workspace> + <tiling>: the
    unused GM workspace pointer stays as a positional placeholder between
    the output and the tiling parameter so the trailing tiling pointer arg
    stays aligned (same contract as lru_cache_v2 /
    dense_lightning_indexer_softmax_lse_v2);
  * the torch wrapper, precision gates and platform prints of the Stage-4
    file are dropped (hosted by op_host / aclnn instead).

"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.tilingkey import TilingKeyField


# ================================================================
# Tile geometry — TilingKey-driven compile-time constants.
#
# pl.TileType shape is a compile-time contract (constant integer list),
# and the UB address map / load_align strides are baked from it, so the
# tile values cannot ride TilingData (runtime) — they ride the
# TilingKey: each key compiles a dedicated kernel instance with the
# values folded to constants (docs tiling_result_transfer.md: "Tile模
# 板、数据排布等有限的离散模式"). Shared contract with
# op_host/lrn_tiling.cpp (LRN_HW_TILE / LRN_C_CHUNK / LRN_R_TIERS):
#
#   * HwTile (candidates declared here, currently [128] for a5) — tile
#     row width = fp16 lanes per VF register (256B / 2B). The kernel
#     cannot query the vreg width itself (see the Q15 note below), so
#     the host carries the same literal (LRN_HW_TILE in
#     op_host/lrn_tiling.cpp) — same-literal contract as CChunk; change
#     BOTH sides together when porting to another platform. The
#     packed-key membership validation (GET_TPL_TILING_KEY ->
#     INVALID_TILING_KEY) fail-closes any one-sided drift.
#   * CChunk (64) — output-channel rows per task, a scheduling choice:
#     more (smaller) tasks split better across cores (multi-core first)
#     rather than maximizing the UB held per task.
#   * RTier ({0,1,2,4,8,16,32,48}) — halo tier: the host picks the
#     smallest candidate >= the ACTUAL depth_radius, so the per-instance
#     x-slot reservation is c_chunk + 2*RTier rows (64..160) instead of
#     the worst case c_chunk + 2*48. The window MATH still uses the
#     exact runtime r from TilingData (clipping unchanged); only the
#     buffer reservation is tiered.
#
# The kernel's task-grid arithmetic uses these constants directly (no
# TilingData tile fields — single source via the key), and the VF chains
# receive them as unannotated literal parameters folded per instance
# (FA tilingkey sample's need_mask precedent: call-site literal args
# trace to constants, so in-VF arithmetic like k * hw_tile lowers
# identically to the former module-global form).
#
# Note: the kernel CANNOT query the vreg width itself — the PyPTO-Pro
# platform surface (pypto_pro.runtime.platform -> pypto_impl bindings)
# exposes only arch/core counts and on-chip buffer capacities
# (GetMemoryLimitForArch), no register width (tilefwk AivCore holds
# vector_reg_width_ but no pybind getter). Ops::Base::GetVRegSize() is
# an op_host C++ API. The TilingKey is therefore the sanctioned channel
# for the platform vreg width (docs tiling_result_transfer.md).
# ================================================================


# ================================================================
# datatype-schema placeholder dtype (module level so the Python
# annotation expressions evaluate at def time; at parse time the
# datatype constant of the active launch key overrides it)
# ================================================================
io_dtype = pl.DT_FP16


# ================================================================
# Tiling data (mirrors LRNTilingData in op_host/lrn_tiling.h)
# ================================================================
@dataclass
class LRNTiling:
    n: int       # batch N
    c: int       # channels C
    hw: int      # H*W merged columns
    r: int       # depth_radius (exact runtime value for the window clipping)
    bias: float
    alpha: float
    beta: float


# ================================================================
# TilingKey — tile-template selection (compile-time per instance)
# ================================================================
class LRNTilingKey:
    # Field names are usable directly inside the kernel body as
    # per-instance compile-time constants (and fold to literals at VF
    # call sites). The fp16/fp32 dtype matrix is dispatched through the
    # datatype schema (per-signature ORIG_DTYPE macros of the AscendC
    # binary build), NOT through this key — the build enumerates one
    # kernel instance per (key, datatype) combination: 8 tiers x 2
    # dtypes = 16 instances (packed keys 0..7; HwTile/CChunk contribute
    # 0 bits as single candidates).
    HwTile = TilingKeyField(bits=2, values=[128])  # fp16 lanes per VF register
    CChunk = TilingKeyField(bits=2, values=[64])   # output-channel rows per task
    RTier = TilingKeyField(bits=3, values=[0, 1, 2, 4, 8, 16, 32, 48])


# ================================================================
# VF register-domain compute chains (verbatim from the Stage-4 kernel)
# ================================================================
@pl.vector_function
def lrn_fp16_vf(x_tile, y32_tile, c_eff: pl.DT_INT64, g_rows: pl.DT_INT64,
                off: pl.DT_INT64, r: pl.DT_INT64, hw_tile,
                alpha: pl.DT_FP32, bias: pl.DT_FP32, beta: pl.DT_FP32):
    """fp16 combo VF chain (widen -> fp32 compute -> INTLV_B32 staging store).

    tile row u maps to GM channel row g_lo + u. For output row i (GM row
    c0 + i) the clipped window is j in [j_lo, j_hi] with
        j_lo = max(0, off - i), j_hi = min(2r, g_rows + off - i - 1),
        u = i + j - off  (whole-row-multiple offset, u in [0, g_rows)).
    The window centre (GM row c0 + i) is tile row u_c = i + r - off; the
    design freezes the "reload once outside the loop" variant (DESIGN.md §1
    centre-row note) — same address, same data, simplest control flow.

    hw_tile is an unannotated literal parameter (TilingKey constant folded
    at the call site — the FA tilingkey sample's need_mask precedent), so
    the addressing arithmetic below is compile-time per instance.
    """
    # mask width follows the *source* dtype of each consuming instruction:
    # astype's preg is b16 (fp16 source reg); fp32-domain arithmetic and the
    # INTLV store's preg are b32 (fp32 half regs). (MW-R02)
    m16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    m32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    neg_beta = -beta
    two_r = 2 * r
    for i in pl.range(0, c_eff):
        # clipped window in tile-row space: u in [u_lo, u_hi], equivalent to
        # j in [j_lo, j_hi] via u = i + j - off (DESIGN.md §1; the golden's
        # clipped window term-by-term)
        u_lo = pl.max(0, i - off)
        u_hi = pl.min(i + two_r - off, g_rows - 1)
        acc_e = vf.full(0.0, m32, dtype=pl.DT_FP32)
        acc_o = vf.full(0.0, m32, dtype=pl.DT_FP32)
        # The dynamic window start is absorbed into an OUTER-scalar base
        # (base = u_lo * hw_tile, the layernorm sample's proven addressing
        # form: load_align(in_tile, base + r * LANES) with r a zero-start
        # inner loop variable). A dynamic inner-loop START miscompiles the
        # in-loop load addressing (backend lowers vlds with a runtime loop
        # base incorrectly — verified by an exact prefix-sum model), and an
        # in-loop scalar vmuls operand is rejected ("must be hoisted"), so
        # the constant-zero-start inner loop over k with an outer base is
        # the only verified-correct form. Mathematically the window is
        # still the exact clipped [u_lo, u_hi] term-by-term (no differencing).
        base = u_lo * hw_tile
        wlen = u_hi - u_lo + 1
        for k in pl.range(0, wlen):
            reg16 = vf.load_align(x_tile, base + k * hw_tile)  # 128 fp16/row
            # widen BEFORE squaring (65504^2 overflows fp16); one astype call
            # yields only 64 fp32, so ZERO (even elems [2k]) + ONE (odd elems
            # [2k+1]) together cover the full 128-fp16 register. (PR-R08/R09)
            f_e = vf.astype(reg16, m16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
            f_o = vf.astype(reg16, m16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
            # vf.* results must be bound to a variable before reuse (no
            # nested expressions in the statement form)
            sq_e = vf.mul(f_e, f_e, m32)
            sq_o = vf.mul(f_o, f_o, m32)
            acc_e = vf.add(acc_e, sq_e, m32)
            acc_o = vf.add(acc_o, sq_o, m32)
        # centre row x itself, reloaded once outside the window loop
        u_c = i + r - off
        creg = vf.load_align(x_tile, u_c * hw_tile)
        x_e = vf.astype(creg, m16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        x_o = vf.astype(creg, m16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        # norm = bias + alpha * S  (fp32 scalars; norm > 0 by bias > 0)
        sa_e = vf.muls(acc_e, alpha, m32)
        sa_o = vf.muls(acc_o, alpha, m32)
        norm_e = vf.adds(sa_e, bias, m32)
        norm_o = vf.adds(sa_o, bias, m32)
        # pow = norm^(-beta) = exp(-beta * ln(norm)) — generic stable path
        ln_e = vf.ln(norm_e, m32)
        ln_o = vf.ln(norm_o, m32)
        nb_e = vf.muls(ln_e, neg_beta, m32)
        nb_o = vf.muls(ln_o, neg_beta, m32)
        pow_e = vf.exp(nb_e, m32)
        pow_o = vf.exp(nb_o, m32)
        y_e = vf.mul(x_e, pow_e, m32)
        y_o = vf.mul(x_o, pow_o, m32)
        # (even, odd) halves interleaved back into 128 contiguous fp32 — the
        # exact inverse of the ZERO/ONE widen semantics. (PR-R10)
        vf.store_align(y32_tile + i * hw_tile, y_e, y_o, m32,
                       dist=pl.StoreDist.INTLV_B32)


@pl.vector_function
def lrn_fp32_vf(x_tile, y32_tile, c_eff: pl.DT_INT64, g_rows: pl.DT_INT64,
                off: pl.DT_INT64, r: pl.DT_INT64, hw_tile, n_q, lanes_f32,
                alpha: pl.DT_FP32, bias: pl.DT_FP32, beta: pl.DT_FP32):
    """fp32 combo VF chain — no astype, no INTLV, no pl.cast (DESIGN.md §1).

    hw_tile / n_q / lanes_f32 are unannotated literal parameters (TilingKey
    constants folded at the call site) — compile-time per instance.
    """
    m32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    neg_beta = -beta
    two_r = 2 * r
    for i in pl.range(0, c_eff):
        # clipped window in tile-row space (same as fp16; see the addressing
        # note in lrn_fp16_vf — outer base + zero-start inner loop)
        u_lo = pl.max(0, i - off)
        u_hi = pl.min(i + two_r - off, g_rows - 1)
        base = u_lo * hw_tile
        wlen = u_hi - u_lo + 1
        u_c = i + r - off
        for q in pl.range(0, n_q):                # column-block q in [0, n_q)
            acc = vf.full(0.0, m32, dtype=pl.DT_FP32)
            for k in pl.range(0, wlen):
                f = vf.load_align(x_tile, base + k * hw_tile + q * lanes_f32)
                sq = vf.mul(f, f, m32)
                acc = vf.add(acc, sq, m32)
            x_c = vf.load_align(x_tile, u_c * hw_tile + q * lanes_f32)
            sa = vf.muls(acc, alpha, m32)
            norm = vf.adds(sa, bias, m32)
            ln_v = vf.ln(norm, m32)
            nb = vf.muls(ln_v, neg_beta, m32)
            pw = vf.exp(nb, m32)
            y_v = vf.mul(x_c, pw, m32)
            vf.store_align(y32_tile + i * hw_tile + q * lanes_f32, y_v, m32)


# ================================================================
# Kernel — dynamic rank: raw pointers, shapes/attrs come from tiling
# ================================================================

@pl.jit(
    auto_mutex=True,
    tiling_key=LRNTilingKey,
    datatype={"x": "io_dtype", "y": "io_dtype"},
)
def lrn(
    x: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: LRNTiling,
):
    """LRN forward (TF across-channels semantics) in one section_vector.

    The ``workspace`` parameter is part of the GE task arg layout
    (``<tensors> + <workspace> + <tiling>``); LRN has no custom workspace,
    so the pointer is an unused positional placeholder that keeps the
    trailing tiling pointer arg aligned.
    """
    n_dim = tiling.n
    c_dim = tiling.c
    hw = tiling.hw
    r = tiling.r
    bias = tiling.bias
    alpha = tiling.alpha
    beta = tiling.beta
    # H*W merge INSIDE the kernel (host does zero view/reshape) — the merged
    # dims are memory-contiguous (stride(H) = W, stride(W) = 1); the view
    # dtype comes from the compile-time io_dtype specialization.
    x2d = pl.make_tensor(x, [n_dim * c_dim, hw], [hw, 1], dtype=io_dtype)
    y2d = pl.make_tensor(y, [n_dim * c_dim, hw], [hw, 1], dtype=io_dtype)

    # ---- tile geometry from the TilingKey (per-instance constants) ----
    hw_tile = HwTile                       # fp16 lanes per VF register (a5: 128)
    lanes_f32 = hw_tile // 2               # fp32 lanes per VF register (vreg bytes / 4):
                                           #   fp16 is 2 B and fp32 is 4 B per element, so
                                           #   exactly half the fp16 lane count (a5: 64)
    c_chunk = CChunk                       # 64 (output-channel rows per task)
    c_ext = c_chunk + 2 * RTier            # x tile physical rows: 64..160,
                                           #   shrunk to this instance's halo tier
                                           #   (worst case was c_chunk + 2*48 = 160)
    n_q = hw_tile // lanes_f32             # 2, fp32 combo column blocks per row:
                                           #   fp16 combo = 1 x full-register row; fp32 combo
                                           #   = n_q x lanes_f32-lane blocks; any even vreg
                                           #   width keeps n_q = 2
                                           #   row pitch bytes: fp16 hw_tile*2, fp32 hw_tile*4 — 32B aligned

    # ---- Tile declarations + UB(Vec) address tables (DESIGN.md §2/§3, outside the section) ----
    if io_dtype == pl.DT_FP16:
        # fp16 canonical combo: x16 rotating x2 + y32 resident x1 + y16 rotating x2
        #   x16_group [c_ext,128] FP16 x2 rotating
        #   y32_group [  64,128] FP32 x1 resident
        #   y16_group [  64,128] FP16 x2 rotating
        #   total = 2*(c_ext*256) + 32768 + 2*16384 (a5 tier48: 147456 B / 253952 B)
        x16_slot_bytes = c_ext * hw_tile * 2
        y32_slot_bytes = c_chunk * hw_tile * 4
        y16_slot_bytes = c_chunk * hw_tile * 2
        va16_x0 = 0x00000
        va16_x1 = va16_x0 + x16_slot_bytes
        va16_y32 = va16_x1 + x16_slot_bytes
        va16_y0 = va16_y32 + y32_slot_bytes
        va16_y1 = va16_y0 + y16_slot_bytes
        x16_type = pl.TileType(shape=[c_ext, hw_tile], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1])
        y32_type = pl.TileType(shape=[c_chunk, hw_tile], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1])
        y16_type = pl.TileType(shape=[c_chunk, hw_tile], dtype=pl.DT_FP16,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1])
        x_group = pl.make_tile_group(type=x16_type,
                                     addrs=[va16_x0, va16_x1], mutex_ids=[0, 1])
        y32_group = pl.make_tile_group(type=y32_type, addrs=[va16_y32],
                                       mutex_ids=[2])
        y16_group = pl.make_tile_group(type=y16_type,
                                       addrs=[va16_y0, va16_y1],
                                       mutex_ids=[3, 4])
    else:
        # fp32 variant combo: x32 rotating x2 + y32 rotating x2
        #   x32_group [c_ext,128] FP32 x2 rotating
        #   y32_group [  64,128] FP32 x2 rotating
        #   total = 2*(c_ext*512) + 2*32768 (a5 tier48: 229376 B / 253952 B)
        x32_slot_bytes = c_ext * hw_tile * 4
        y32_out_slot_bytes = c_chunk * hw_tile * 4
        va32_x0 = 0x00000
        va32_x1 = va32_x0 + x32_slot_bytes
        va32_y0 = va32_x1 + x32_slot_bytes
        va32_y1 = va32_y0 + y32_out_slot_bytes
        x32_type = pl.TileType(shape=[c_ext, hw_tile], dtype=pl.DT_FP32,
                               target_memory=pl.MemorySpace.Vec,
                               valid_shape=[-1, -1])
        y_out_type = pl.TileType(shape=[c_chunk, hw_tile], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec,
                                 valid_shape=[-1, -1])
        x_group = pl.make_tile_group(type=x32_type,
                                     addrs=[va32_x0, va32_x1], mutex_ids=[0, 1])
        y_out_group = pl.make_tile_group(type=y_out_type,
                                         addrs=[va32_y0, va32_y1],
                                         mutex_ids=[2, 3])

    with pl.section_vector():
        core_id = pl.get_block_idx()
        num_cores = pl.get_block_num()

        # task-grid arithmetic uses the key-frozen compile-time constants
        # (single source — no TilingData tile fields)
        c_blks = (c_dim + c_chunk - 1) // c_chunk
        hw_blks = (hw + hw_tile - 1) // hw_tile
        total_tasks = n_dim * c_blks * hw_blks

        for task_id in pl.range(core_id, total_tasks, num_cores):
            # task grid unravel: n outer, c_blk middle, hw_blk innermost
            # (GM access locality within one c block)
            n_idx = task_id // (c_blks * hw_blks)
            rem = task_id % (c_blks * hw_blks)
            c_blk = rem // hw_blks
            hw_blk = rem % hw_blks
            c0 = c_blk * c_chunk
            hw_off = hw_blk * hw_tile
            # halo window actually loaded (clipped to [0, C))
            g_lo = pl.max(0, c0 - r)
            g_hi = pl.min(c_dim, c0 + c_chunk + r)
            g_rows = g_hi - g_lo
            # per-task valid extents (all min/max clamps — TV-R01)
            c_eff = pl.min(c_chunk, c_dim - c0)
            hw_eff = pl.min(hw_tile, hw - hw_off)
            off = g_lo - c0 + r

            x_slot = x_group.next()
            pl.set_validshape(x_slot, [g_rows, hw_eff])     # set BEFORE load
            pl.load(x_slot, x2d, [n_idx * c_dim + g_lo, hw_off])

            if io_dtype == pl.DT_FP16:
                y32_slot = y32_group.current()              # resident single slot
                pl.set_validshape(y32_slot, [c_eff, hw_eff])
                lrn_fp16_vf(x_slot, y32_slot, c_eff, g_rows, off, r, hw_tile,
                            alpha, bias, beta)
                y16_slot = y16_group.next()
                pl.set_validshape(y16_slot, [c_eff, hw_eff])  # same as y32 (cast)
                pl.cast(y16_slot, y32_slot, mode=pl.RoundMode.CAST_RINT)
                pl.store(y2d, y16_slot, [n_idx * c_dim + c0, hw_off])
            else:
                y_slot = y_out_group.next()
                pl.set_validshape(y_slot, [c_eff, hw_eff])
                lrn_fp32_vf(x_slot, y_slot, c_eff, g_rows, off, r, hw_tile,
                            n_q, lanes_f32, alpha, bias, beta)
                pl.store(y2d, y_slot, [n_idx * c_dim + c0, hw_off])
    return
