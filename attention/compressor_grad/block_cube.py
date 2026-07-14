#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""block_cube: combined cube kernel for compressor_grad backward.

Single @pl.jit kernel with ONE section_cube that computes all three matmul
backwards from golden stage D (compressor_grad_golden.py lines 527-529):

    d_wkv   [C, H] = d_new_kv.T   [C, T] @ x_flat [T, H]
    d_wgate [C, H] = d_new_score.T [C, T] @ x_flat [T, H]
    d_x     [T, H] = d_new_kv [T, C] @ wkv   [C, H]
                       + d_new_score [T, C] @ wgate [C, H]

Transpose of d_new_kv / d_new_score for the d_wkv / d_wgate matmuls is
materialized INSIDE the kernel via ``pl.load(..., order=[1, 0])``
transpose-loads -- the host does NOT pre-transpose (no ``.t().contiguous()``).
The receiving Mat tile groups (a_l1 in d_wkv / d_wgate) are declared
``layout=pl.ZN`` (k_tn convention in test_load_transpose_convention.py) so the
on-chip buffer holds the transposed [M, K] = [C_slice, T_slice] form the cube
expects.  Because ``order=[1, 0]`` makes the compiler relabel its source GM
view as ``Layout::DN`` (column-major), two independent GM views of each pointer
are created -- one for the transpose-load path (d_wkv / d_wgate) and one for
the normal-load path (d_x) -- so each view's layout interpretation stays
consistent for all of its loads.

Sequence-length / batch-size generalization:
    T = B * S is NOT a compile-time constant.  It flows in at launch time
    through TilingData.tokenSize / hiddenSize / headDim, so the SAME compiled
    kernel covers B in [1, 64+] and S in {128, 4096, 8192, 16384} (and beyond,
    as long as tile boundaries stay aligned).  For T > wkv_tile_k * num_cores
    (B>1 with S=128, or B=1 with S>=4096) the per-core K range of d_wkv /
    d_wgate grows beyond one wkv_tile_k tile, so an inner K-tile loop with
    matmul / matmul_acc (phase Partial/Final) accumulates the per-core K
    partial before the cross-core atomic add to GM.  For d_x the M=T axis is
    tiled by dx_tile_m so the same N-split kernel scales to arbitrary T.

Multi-core strategy:
    d_wkv / d_wgate: K-split -- each core owns a K=T/num_cores slice, tiled
                     by wkv_tile_k and accumulated into one L0C tile; cross-core
                     accumulation via atomic add to GM.  Output MUST be zero-init.
    d_x:             N-split -- each core owns H/num_cores columns; M=T tiled by
                     dx_tile_m, K=C tiled into K-blocks, two matmuls accumulated
                     into the same L0C.  No atomic add needed.

The three helpers run SEQUENTIALLY inside the single section_cube, separated by
``pl.system.bar_all()``.  An attempt to run d_wkv / d_wgate in parallel (with
non-overlapping L1/L0A/L0B/L0C ranges and no bar_all between them) broke
precision: the cube unit's hardware pipeline overlaps the FIX store of the
first helper with the MTE2 load of the second even when their on-chip buffers
do not overlap, and auto_mutex alone (different mutex_ids) cannot synchronize
across this implicit pipeline interaction.  Hence the conservative bar_all.
"""

import os
import logging
from dataclasses import dataclass

import torch
import torch_npu

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
from pypto_pro.runtime.platform import get_platform_info


# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

LOG_FORMATTER = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]'
)
LOG_HANDLER = logging.StreamHandler()
LOG_HANDLER.setFormatter(LOG_FORMATTER)
logger.handlers.clear()
logger.addHandler(LOG_HANDLER)


def get_device() -> str:
    """Get the appropriate device string for computation.

    Returns:
        Device string (e.g., "npu:0").
    """
    cur_device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(int(cur_device_id))
    device = f"npu:{cur_device_id}"
    return device


def compare(t: torch.Tensor, t_ref: torch.Tensor, name, *, atol, rtol, max_error_ratio=0.005, max_error_count=10):
    """
    比较两个张量的差异，超过阈值时打印错误点并抛出断言错误
    Args:
        t: 待比较张量
        t_ref: 参考张量
        name: 张量名称（用于日志）
        atol: 绝对容差
        rtol: 相对容差
        max_error_ratio: 误差点占总元素数的最大比例
        max_error_count: 显示的最大误差点数量（同时也是误差点阈值的上限）
    """
    def check_is_nan_inf():
        # ========== 核心新增：检测t中的NaN和Inf并直接报错 ==========
        # 1. 检测NaN
        nan_mask = torch.isnan(t)
        nan_count = nan_mask.sum().item()

        # 2. 检测Inf（包含+Inf和-Inf）
        inf_mask = torch.isinf(t)
        inf_count = inf_mask.sum().item()

        # 若存在NaN或Inf，拼接错误信息并报错
        if nan_count > 0 or inf_count > 0:
            error_msg = f"\n========== 张量 {name} 检测到非法值（禁止存在NaN/Inf）=========="

            # 打印NaN信息
            if nan_count > 0:
                nan_positions = torch.nonzero(nan_mask, as_tuple=False)
                show_nan_count = min(nan_count, max_error_count)
                error_msg += f"\n- NaN数量：{nan_count}，前 {show_nan_count} 个位置："
                for i in range(show_nan_count):
                    pos_tuple = tuple(p.item() for p in nan_positions[i])
                    error_msg += f"\n  位置 {pos_tuple}"

            # 打印Inf信息（区分+Inf/-Inf）
            if inf_count > 0:
                inf_positions = torch.nonzero(inf_mask, as_tuple=False)
                show_inf_count = min(inf_count, max_error_count)
                error_msg += f"\n- Inf数量：{inf_count}，前 {show_inf_count} 个位置（值类型）："
                for i in range(show_inf_count):
                    pos = inf_positions[i]
                    pos_tuple = tuple(p.item() for p in pos)
                    inf_val = t[pos_tuple].item()
                    inf_type = "+Inf" if inf_val == float('inf') else "-Inf"
                    error_msg += f"\n  位置 {pos_tuple}：{inf_type}"
            error_msg += "\n" + "=" * 80 + "\n"

            # 抛出断言错误，终止函数执行
            assert False, error_msg

    # check 是否是nan 或 inf
    check_is_nan_inf()

    # 先验证张量的基本属性一致
    assert t.shape == t_ref.shape, f"张量形状不一致：t.shape={t.shape}, t_ref.shape={t_ref.shape}"
    assert t.dtype == t_ref.dtype, f"张量数据类型不一致：t.dtype={t.dtype}, t_ref.dtype={t_ref.dtype}"
    assert t.device == t_ref.device, f"张量设备不一致：t.device={t.device}, t_ref.device={t_ref.device}"

    # 计算误差点数量的阈值（取比例计算值和最大数量的较小值）
    error_count_threshold = round(max_error_ratio * t_ref.numel())

    # 计算误差掩码（超过阈值的位置为True）
    diff_abs = (t - t_ref).abs()
    tolerance = atol + rtol * t_ref.abs()
    diff_mask = diff_abs > tolerance
    error_count = diff_mask.sum().item()

    # 计算最大误差和其位置
    max_diff, flat_max_pos = torch.max(diff_abs.flatten(), dim=0)
    max_pos = torch.unravel_index(flat_max_pos, t.shape)
    max_pos = tuple(idx.item() for idx in max_pos)

    # 打印错误点的逻辑（如果有误差点）
    if error_count > 0:
        print(f"\n========== 张量 {name} 存在 {error_count} 个误差点（阈值：{error_count_threshold}）==========")

        # 获取所有误差点的位置
        error_positions = torch.nonzero(diff_mask, as_tuple=False)  # shape: [error_count, dims]

        # 限制显示的误差点数量（避免数据量过大）
        show_count = min(error_count, max_error_count)
        print(f"显示前 {show_count} 个误差点（位置 | 待比较值 | 参考值 | 绝对误差 | 允许阈值）：")

        # 遍历前N个误差点打印详细信息
        for i in range(show_count):
            pos = error_positions[i]

            # 转换为元组格式的位置（如 (0, 2, 3)）
            pos_tuple = tuple(p.item() for p in pos)

            # 获取对应位置的数值
            t_val = t[pos_tuple].item()
            t_ref_val = t_ref[pos_tuple].item()
            diff_val = diff_abs[pos_tuple].item()
            tol_val = tolerance[pos_tuple].item()

            # 格式化输出，保留足够小数位
            print(f"  位置 {pos_tuple}: {t_val:.8f} vs {t_ref_val:.8f} | 误差={diff_val:.8f} | 阈值={tol_val:.8f}")

        # 打印最大误差点
        print(f"\n最大误差点：位置 {max_pos} | 误差={max_diff.item():.8f} | 阈值={tolerance[max_pos].item():.8f}")
        print("=" * 80 + "\n")

    # 断言误差点数量不超过阈值
    assert error_count <= error_count_threshold, \
        (f"compare fail: {name}, max diff: {max_diff.item():.8f} at {max_pos}, "
         f"error_count: {error_count}, error_count_threshold: {error_count_threshold}")


# ================================================================
#  TilingKey — compile-time specialization schema
# ================================================================
class CompressorTilingKey:
    """Compile-time choices consumed by ``block_cube_kernel``.

    Coff     : overlap factor (1 or 2) -> C = Coff * head_dim
    Layout   : 0 = BSH (x is [B, S, H]), 1 = TH (x is [T, H])
    DataType : 0 = BF16, 1 = FP16
    """
    Coff     = TilingKeyField(bits=2, values=[1, 2])
    Layout   = TilingKeyField(bits=1, values=[0, 1])
    DataType = TilingKeyField(bits=2, values=[0, 1])

    def is_valid(self, key):
        coff, layout, dataType = key
        return coff in (1, 2) and layout in (0, 1) and dataType in (0, 1)


# ================================================================
#  TilingData — runtime tiling parameters (passed as last kernel arg)
# ================================================================
@dataclass
class CompressorTiling:
    batchSize:  int = 1
    cmpRatio:   int = 4        # compression ratio, runtime (typical 2~128)
    tokenSize:  int = 0       # T = B * S, runtime
    hiddenSize: int = 0       # H, runtime
    headDim:    int = 0       # D, runtime (used for c_size = Coff * headDim)
    totalBlks:  int = 0


# ════════════════════════════════════════════════════════════════════════════
# Helper 1: d_wkv  (mutex_ids 0-9)
# ════════════════════════════════════════════════════════════════════════════
def d_wkv(a, b, out, c_size, t_size, h_size, io_dtype) -> None:
    """Matmul backward for wkv weight gradient: d_wkv[C, H] = a[C, T] @ b[T, H].

    ``a`` is a GM view of d_new_kv in its natural [T, C] layout (NOT
    host-transposed).  The required [C, T] left-operand form is materialized
    INSIDE this helper via a transpose-load: ``pl.load(a_l1, a, [t_off, c_off],
    order=[1, 0])`` reads a [T_slice, C_slice] window from GM into the ``a_l1``
    Mat tile group whose TileType shape is [wkv_tile_m, wkv_tile_k] =
    [C_slice, T_slice] with ``layout=pl.ZN``.  The GM offsets are therefore
    given in GM axis order -- [T axis, C axis] -- while the loaded tile's
    logical shape is the transpose, matching the k_tn convention in
    test_load_transpose_convention.py / test_load_transpose_tile_group.py
    (need_transpose=True -> Mat layout=ZN, Mat shape=reverse(GM) = [M, K]).

    K-split multi-core: K=T is split across num_cores cores, each core owns a
    contiguous K slice of size ``t_size // num_cores``.  When the per-core K
    slice exceeds a single K-tile (B > 1), the slice is tiled and accumulated
    into one L0C tile via matmul / matmul_acc (phase Partial/Final).  Cross-core
    accumulation is done by atomic add to GM, so the output GM tensor MUST be
    zero-init.

    The three cube tile sizes (WKV_TILE_M/K/N) and the b_l1 base address are
    compile-time constants derived from the Coff TilingKey field (see the
    ``if Coff == 1 / else`` block below); each coff value compiles to a
    dedicated kernel whose M/K/N tile shapes are tuned to keep a_l1 / a_left
    within L1 / L0A capacity while covering the M axis (C = coff * head_dim)
    naturally.  The b_l1 base address is computed as
    ``4 * WKV_TILE_M * WKV_TILE_K`` (a_l1 is double-buffered: 2 slots * M*K*2
    bytes for BF16/FP16), which the parser folds to a compile-time constant.
    """
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    k_per_core = t_size // num_cores
    k_start = core_id * k_per_core

    # ── Coff-dependent tiling (compile-time constants per TilingKey) ──
    # Tile shapes chosen so a_l1 / a_left stay within L1 / L0A capacity;
    # see docstring above for the per-coff sizing rationale.
    if Coff == 1:
        wkv_tile_m = 128
        wkv_tile_k = 64
        wkv_tile_n = 128
    else:  # Coff == 2
        wkv_tile_m = 256
        wkv_tile_k = 32
        wkv_tile_n = 64

    # L1 (Mat) address layout -- d_wkv uses the lower half of L1/L0A/L0B/L0C.
    # Even though the three helpers run sequentially (separated by bar_all),
    # d_wkv and d_wgate are given non-overlapping on-chip ranges so that a
    # future tightening of the inter-helper barrier (e.g. replacing bar_all
    # with finer-grained sync_src/sync_dst) can overlap them without further
    # address changes.  Coff=2 sizes (most constraining): a_l1 slot =
    # 256*32*2 = 16 KB (x2 = 32 KB), b_l1 slot = 32*64*2 = 4 KB (x2 = 8 KB),
    # a_left slot = 16 KB (x2 = 32 KB), b_right slot = 4 KB (x2 = 8 KB),
    # acc slot = 256*64*4 = 64 KB (x2 = 128 KB).
    #
    #   d_wkv:   a_l1 [0x00000, 0x08000)  b_l1 [0x08000, 0x0A000)
    #            a_left [0x0000, 0x8000)  b_right [0x0000, 0x2000)
    #            acc [0x00000, 0x20000)
    #   d_wgate: a_l1 [0x0A000, 0x12000)  b_l1 [0x12000, 0x14000)
    #            a_left [0x8000, 0x10000) b_right [0x2000, 0x4000)
    #            acc [0x20000, 0x40000)
    #
    # Total L1 = 80 KB, total L0A = 64 KB (both within A5 capacity).  Coff=1
    # uses smaller tiles (a_l1 slot = 128*64*2 = 16 KB, b_l1 slot =
    # 128*128*2 = 32 KB) so the same address map applies with the
    # Coff-conditional b_l1 base in d_wgate adjusted accordingly.
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[wkv_tile_m, wkv_tile_k], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=0x00000, mutex_ids=[0, 1],
    )
    # b_l1 base = end of a_l1's double-buffered range (2 slots * M*K*2 bytes).
    # Both coff values produce a_l1 span = 32 KB = 0x8000, so a single literal
    # works; d_wgate's b_l1 (which DOES depend on coff) uses a Coff-conditional.
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[wkv_tile_k, wkv_tile_n], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Mat),
        addrs=0x08000, mutex_ids=[2, 3],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[wkv_tile_m, wkv_tile_k], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Left),
        addrs=0x00000, mutex_ids=[4, 5],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[wkv_tile_k, wkv_tile_n], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Right),
        addrs=0x00000, mutex_ids=[6, 7],
    )
    # L0C accumulator: fractal=1024 is mandatory for FP32 K-dim accumulation;
    # harmless for the single-K-tile path.  Two mutex_ids preserve the original
    # double-buffer behavior for the single-tile (B==1) path.
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[wkv_tile_m, wkv_tile_n], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, fractal=1024),
        addrs=0x00000, mutex_ids=[8, 9],
    )

    # When k_per_core > wkv_tile_k the per-core K slice spans multiple K-tiles
    # and must be K-accumulated in L0C before the cross-core atomic store.
    needs_k_acc = k_per_core > wkv_tile_k
    if needs_k_acc:
        pl.system.set_mm_layout_transform(enabled=True)

    for mi in pl.range(0, c_size, wkv_tile_m):
        for ni in pl.range(0, h_size, wkv_tile_n):
            if needs_k_acc:
                ac = acc.current()
                # K-tile accumulation over this core's K slice [k_start, k_start+k_per_core).
                # ``a`` is GM d_new_kv[T, C]; transpose-load (order=[1, 0]) delivers
                # the [C, T] left-operand -- see docstring above for the convention.
                for ki_off in pl.range(0, k_per_core, wkv_tile_k):
                    ki = k_start + ki_off
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    al = a_left.next()
                    br = b_right.next()
                    pl.load(cur_a, a, [ki, mi], order=[1, 0])
                    pl.load(cur_b, b, [ki, ni])
                    pl.move(al, cur_a)
                    pl.move(br, cur_b)
                    is_first = (ki_off == 0)
                    is_last = (ki_off + wkv_tile_k >= k_per_core)
                    if is_first:
                        pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                    elif is_last:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                    else:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                pl.store(out, ac, [mi, ni], atomic=pl.AtomicType.AtomicAdd,
                         phase=pl.STPhase.Final)
            else:
                # Single K-tile path (B == 1): no K-accumulation, plain matmul
                # followed by atomic add.  Same transpose-load convention as above.
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                al = a_left.next()
                br = b_right.next()
                ac = acc.next()
                pl.load(cur_a, a, [k_start, mi], order=[1, 0])
                pl.load(cur_b, b, [k_start, ni])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(out, ac, [mi, ni], atomic=pl.AtomicType.AtomicAdd)

    if needs_k_acc:
        pl.system.set_mm_layout_transform(enabled=False)


# ════════════════════════════════════════════════════════════════════════════
# Helper 2: d_wgate  (mutex_ids 10-19)
# ════════════════════════════════════════════════════════════════════════════
def d_wgate(a, b, out, c_size, t_size, h_size, io_dtype) -> None:
    """Matmul backward for wgate weight gradient: d_wgate[C, H] = a[C, T] @ b[T, H].

    ``a`` is a GM view of d_new_score in its natural [T, C] layout (NOT
    host-transposed); the [C, T] left-operand form is materialized on-chip via
    ``pl.load(a_l1, a, [t_off, c_off], order=[1, 0])`` into a ZN Mat tile of
    logical shape [wkv_tile_m, wkv_tile_k] = [C_slice, T_slice].  See ``d_wkv``
    for the full transpose-load convention.

    K-split multi-core: identical strategy to d_wkv.  Output MUST be zero-init.
    Tile sizes and b_l1 base address are the same coff-derived compile-time
    constants as ``d_wkv`` (see the ``if Coff == 1 / else`` block below).
    """
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    k_per_core = t_size // num_cores
    k_start = core_id * k_per_core

    # ── Coff-dependent tiling (compile-time constants per TilingKey; see d_wkv)
    if Coff == 1:
        wkv_tile_m = 128
        wkv_tile_k = 64
        wkv_tile_n = 128
    else:  # Coff == 2
        wkv_tile_m = 256
        wkv_tile_k = 32
        wkv_tile_n = 64

    # L1 (Mat) address layout -- d_wgate uses the upper half of the on-chip
    # buffers, non-overlapping with d_wkv (see d_wkv for the full map).  d_wgate
    # starts right after d_wkv's b_l1 ends; the a_l1/b_l1 base addresses are
    # Coff-conditional because d_wkv's b_l1 span depends on coff
    # (coff=1: 32 KB, coff=2: 8 KB).  a_left / b_right / acc use fixed offsets
    # that match the non-overlapping L0A / L0B / L0C slots in the map above.
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[wkv_tile_m, wkv_tile_k], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),
        addrs=(0x8000 + 4 * 64 * 128) if Coff == 1 else (0x8000 + 4 * 32 * 64),
        mutex_ids=[10, 11],
    )
    # b_l1 base = d_wgate a_l1 base + a_l1's double-buffered span (4 * M * K).
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[wkv_tile_k, wkv_tile_n], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Mat),
        addrs=((0x8000 + 4 * 64 * 128) + (4 * 128 * 64)) if Coff == 1
        else ((0x8000 + 4 * 32 * 64) + (4 * 256 * 32)),
        mutex_ids=[12, 13],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[wkv_tile_m, wkv_tile_k], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Left),
        addrs=0x08000, mutex_ids=[14, 15],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[wkv_tile_k, wkv_tile_n], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Right),
        addrs=0x02000, mutex_ids=[16, 17],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[wkv_tile_m, wkv_tile_n], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, fractal=1024),
        addrs=0x20000, mutex_ids=[18, 19],
    )

    needs_k_acc = k_per_core > wkv_tile_k
    if needs_k_acc:
        pl.system.set_mm_layout_transform(enabled=True)

    for mi in pl.range(0, c_size, wkv_tile_m):
        for ni in pl.range(0, h_size, wkv_tile_n):
            if needs_k_acc:
                ac = acc.current()
                # K-tile accumulation; same transpose-load convention as d_wkv
                # (``a`` is GM d_new_score[T, C], order=[1, 0] -> [C, T] left-operand).
                for ki_off in pl.range(0, k_per_core, wkv_tile_k):
                    ki = k_start + ki_off
                    cur_a = a_l1.next()
                    cur_b = b_l1.next()
                    al = a_left.next()
                    br = b_right.next()
                    pl.load(cur_a, a, [ki, mi], order=[1, 0])
                    pl.load(cur_b, b, [ki, ni])
                    pl.move(al, cur_a)
                    pl.move(br, cur_b)
                    is_first = (ki_off == 0)
                    is_last = (ki_off + wkv_tile_k >= k_per_core)
                    if is_first:
                        pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                    elif is_last:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
                    else:
                        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                pl.store(out, ac, [mi, ni], atomic=pl.AtomicType.AtomicAdd,
                         phase=pl.STPhase.Final)
            else:
                # Single K-tile path (B == 1): no K-accumulation, plain matmul
                # followed by atomic add.  Same transpose-load convention as above.
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                al = a_left.next()
                br = b_right.next()
                ac = acc.next()
                pl.load(cur_a, a, [k_start, mi], order=[1, 0])
                pl.load(cur_b, b, [k_start, ni])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(out, ac, [mi, ni], atomic=pl.AtomicType.AtomicAdd)

    if needs_k_acc:
        pl.system.set_mm_layout_transform(enabled=False)


# ════════════════════════════════════════════════════════════════════════════
# Helper 3: d_x  (mutex_ids 20-28)
# ════════════════════════════════════════════════════════════════════════════
def d_x(d_new_kv, wkv, d_new_score, wgate, out, c_size, t_size, h_size, io_dtype) -> None:
    """Matmul backward for input gradient d_x.

        d_x[T, H] = d_new_kv[T, C] @ wkv[C, H] + d_new_score[T, C] @ wgate[C, H]

    N-split multi-core: each core owns ``h_size // num_cores`` columns of N=H.
    M=T is tiled by ``dx_tile_m`` so the same kernel scales from B=1 (single
    M-tile) to arbitrary B (multiple M-tiles).  K-dim (K=C) is tiled by
    ``dx_tile_k`` and the two matmuls are accumulated into the same L0C via
    matmul / matmul_acc.  No atomic add is needed because each output column is
    written by one core.

    Coff-dependent tile sizes and b_l1 base address follow the same
    compile-time-constant scheme as d_wkv (see its docstring); only the
    concrete tile shapes differ (coff=1: 128/128/128, coff=2: 64/64/64).
    """
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    n_per_core = h_size // num_cores
    n_start = core_id * n_per_core

    # ── Coff-dependent tiling (compile-time constants per TilingKey) ──
    # Tile shapes chosen so a_l1 / a_left stay within L1 / L0A capacity;
    # see docstring above for the per-coff sizing rationale.
    if Coff == 1:
        dx_tile_m = 128
        dx_tile_k = 128
        dx_tile_n = 128
    else:  # Coff == 2
        dx_tile_m = 64
        dx_tile_k = 64
        dx_tile_n = 64

    # L1 (Mat) addresses: a_l1 at 0x0; b_l1 right after a_l1's double-buffered
    # range (4 * dx_tile_m * dx_tile_k bytes).  b_l1 base is inlined as a
    # Coff-conditional literal because the parser's addrs= evaluator cannot
    # resolve helper-local variables (only TilingKey fields like Coff).
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[dx_tile_m, dx_tile_k], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[20, 21],
    )
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[dx_tile_k, dx_tile_n], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Mat),
        addrs=(4 * 128 * 128) if Coff == 1 else (4 * 64 * 64), mutex_ids=[22, 23],
    )
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[dx_tile_m, dx_tile_k], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Left),
        addrs=0x00000, mutex_ids=[24, 25],
    )
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[dx_tile_k, dx_tile_n], dtype=io_dtype,
                         target_memory=pl.MemorySpace.Right),
        addrs=0x00000, mutex_ids=[26, 27],
    )
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[dx_tile_m, dx_tile_n], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, fractal=1024),
        addrs=0x00000, mutex_ids=[28],
    )

    pl.system.set_mm_layout_transform(enabled=True)

    # M = T is tiled by dx_tile_m; for B == 1 this is a single M-tile, for B > 1
    # this loops over multiple M-tiles.  N is split across cores (n_start offset).
    for mi in pl.range(0, t_size, dx_tile_m):
        for ni in pl.range(0, n_per_core, dx_tile_n):
            ac = acc.current()

            # ── matmul 1: ac = d_new_kv[mi:mi+M, :] @ wkv[:, n_start+ni : ...] ──
            for ki in pl.range(0, c_size, dx_tile_k):
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                al = a_left.next()
                br = b_right.next()
                pl.load(cur_a, d_new_kv, [mi, ki])
                pl.load(cur_b, wkv, [ki, n_start + ni])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                if ki == 0:
                    pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
                else:
                    pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)

            # ── matmul 2: ac += d_new_score[mi:mi+M, :] @ wgate[:, n_start+ni : ...] ──
            for ki in pl.range(0, c_size, dx_tile_k):
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                al = a_left.next()
                br = b_right.next()
                pl.load(cur_a, d_new_score, [mi, ki])
                pl.load(cur_b, wgate, [ki, n_start + ni])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                if ki < c_size - dx_tile_k:
                    pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
                else:
                    pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)

            pl.store(out, ac, [mi, n_start + ni], phase=pl.STPhase.Final)

    pl.system.set_mm_layout_transform(enabled=False)


# ════════════════════════════════════════════════════════════════════════════
# Single kernel: one section_cube, three sequential helper calls
# ════════════════════════════════════════════════════════════════════════════
@pl.jit(auto_mutex=True, tiling_key=CompressorTilingKey)
def block_cube_kernel(
    x_ptr: pl.Ptr[pl.DT_UINT8],
    d_new_kv_ptr: pl.Ptr[pl.DT_UINT8],
    d_new_score_ptr: pl.Ptr[pl.DT_UINT8],
    wkv_ptr: pl.Ptr[pl.DT_UINT8],
    wgate_ptr: pl.Ptr[pl.DT_UINT8],
    d_x_out_ptr: pl.Ptr[pl.DT_FP32],
    d_wkv_out_ptr: pl.Ptr[pl.DT_FP32],
    d_wgate_out_ptr: pl.Ptr[pl.DT_FP32],
    tiling: CompressorTiling,
):
    """Combined cube kernel for compressor_grad backward (stage D matmuls).

    See the module docstring for the high-level design (transpose-load
    convention, sequence-length generalization, multi-core strategy, sequential
    helper execution with bar_all).  This docstring records only the kernel's
    own contract: argument shapes, TilingKey / TilingData field semantics, and
    the pre-conditions callers must honor.

    Args (all wrapped as ``pl.Ptr[...]`` raw pointers; the body builds tensor
    views via ``pl.make_tensor`` with runtime shapes from ``tiling``):
        x_ptr           : [T, H] (TH) or [B, S, H] contiguous (BSH; reinterpreted
                          as [T, H] inside the kernel for both layouts).
        d_new_kv_ptr    : [T, C] in the natural row-major layout; NOT
                          host-transposed.  Two GM views are built inside:
                          ``d_new_kv_t`` for the transpose-load path (d_wkv)
                          and ``d_new_kv`` for the normal-load path (d_x).
        d_new_score_ptr : same as d_new_kv_ptr, consumed by d_wgate / d_x.
        wkv_ptr, wgate_ptr : [C, H] weights.
        d_x_out_ptr     : [T, H] FP32 output (d_x gradient).
        d_wkv_out_ptr, d_wgate_out_ptr : [C, H] FP32 outputs.  MUST be zero-init
                          before launch (cross-core atomic add accumulates on
                          top).
        tiling          : CompressorTiling (see below).

    TilingKey (compile-time, each value -> dedicated kernel):
        Coff     : 1 or 2  -> C = Coff * head_dim
        Layout   : 0=BSH, 1=TH
        DataType : 0=BF16, 1=FP16

    TilingData (runtime, see CompressorTiling):
        batchSize  : B (informational; T already folds it in via tokenSize)
        cmpRatio   : compression ratio (informational; cube stage-D does not
                     read it -- carried for parity with the golden interface)
        tokenSize  : T = B * S
        hiddenSize : H
        headDim    : D    (used for c_size = Coff * headDim)
    """
    # ── DataType -> io_dtype (compile-time) ──
    if DataType == 0:
        io_dtype = pl.DT_BF16
    else:
        io_dtype = pl.DT_FP16

    # ── Coff (compile-time), head_dim (runtime) -> c_size ──
    c_size = Coff * tiling.headDim

    # ── Runtime shape parameters from TilingData ──
    t_size = tiling.tokenSize        # T = B * S, runtime
    h_size = tiling.hiddenSize       # H, runtime

    # ── Wrap bare pointers into tensor views (shapes use runtime t_size/h_size) ──
    # d_new_kv / d_new_score each get TWO views of the same pointer: the ``_t``
    # variant feeds the transpose-load path (d_wkv / d_wgate), the plain variant
    # feeds the normal-load path (d_x).  See module docstring for why the two
    # views are needed (order=[1,0] relabels its source view as Layout::DN).
    d_new_kv_t = pl.make_tensor(d_new_kv_ptr, [t_size, c_size], [c_size, 1], dtype=io_dtype)
    d_new_score_t = pl.make_tensor(d_new_score_ptr, [t_size, c_size], [c_size, 1], dtype=io_dtype)
    d_new_kv = pl.make_tensor(d_new_kv_ptr, [t_size, c_size], [c_size, 1], dtype=io_dtype)
    d_new_score = pl.make_tensor(d_new_score_ptr, [t_size, c_size], [c_size, 1], dtype=io_dtype)
    wkv = pl.make_tensor(wkv_ptr, [c_size, h_size], [h_size, 1], dtype=io_dtype)
    wgate = pl.make_tensor(wgate_ptr, [c_size, h_size], [h_size, 1], dtype=io_dtype)
    d_wkv_out = pl.make_tensor(d_wkv_out_ptr, [c_size, h_size], [h_size, 1], dtype=pl.DT_FP32)
    d_wgate_out = pl.make_tensor(d_wgate_out_ptr, [c_size, h_size], [h_size, 1], dtype=pl.DT_FP32)

    # x and d_x_out: both BSH (B, S, H) and TH (T, H) are GM-contiguous and
    # share the same (T, H) memory layout (T = B*S), so a single 2-D view works
    # for both Layout values -- mirrors golden's x_flat = x.reshape(T, H).
    x_flat = pl.make_tensor(x_ptr, [t_size, h_size], [h_size, 1], dtype=io_dtype)
    d_x_out = pl.make_tensor(d_x_out_ptr, [t_size, h_size], [h_size, 1], dtype=pl.DT_FP32)

    with pl.section_cube():
        d_wkv(d_new_kv_t, x_flat, d_wkv_out, c_size, t_size, h_size, io_dtype)
        pl.system.bar_all()
        d_wgate(d_new_score_t, x_flat, d_wgate_out, c_size, t_size, h_size, io_dtype)
        pl.system.bar_all()
        d_x(d_new_kv, wkv, d_new_score, wgate, d_x_out, c_size, t_size, h_size, io_dtype)


# ════════════════════════════════════════════════════════════════════════════
# Test cases
# ════════════════════════════════════════════════════════════════════════════

RTOL = 5e-3
ATOL = 5e-2
MAX_ERROR_RATIO = 0.005


def _gen_cube_inputs(seed, grad_scale, weight_scale, c_size, torch_dtype, layout, device,
                     b_size=1, s_size=128, h_size=4096):
    """Generate cube-stage inputs for block_cube_kernel.

    Shapes:
        BSH (layout=0): x is (B, S, H)
        TH  (layout=1): x is (T, H)
        d_new_kv, d_new_score: (T, C)   [both layouts]
        wkv, wgate:            (C, H)   [both layouts]

    ``b_size`` generalizes the batch dimension; T = b_size * s_size.
    """
    torch.manual_seed(seed)
    T = b_size * s_size
    C, H = c_size, h_size
    d_new_kv = (torch.rand((T, C), device=device, dtype=torch.float32) * grad_scale * 2 - grad_scale)
    d_new_score = (torch.rand((T, C), device=device, dtype=torch.float32) * grad_scale * 2 - grad_scale)
    wkv = (torch.rand((C, H), device=device, dtype=torch.float32) * weight_scale * 2 - weight_scale)
    wgate = (torch.rand((C, H), device=device, dtype=torch.float32) * weight_scale * 2 - weight_scale)

    if layout == 0:  # BSH
        x = (torch.rand((b_size, s_size, H), device=device, dtype=torch.float32) * weight_scale * 2 - weight_scale)
    else:  # TH
        x = (torch.rand((T, H), device=device, dtype=torch.float32) * weight_scale * 2 - weight_scale)

    return {
        "d_new_kv": d_new_kv.to(torch_dtype),
        "d_new_score": d_new_score.to(torch_dtype),
        "x": x.to(torch_dtype),
        "wkv": wkv.to(torch_dtype),
        "wgate": wgate.to(torch_dtype),
    }


def _get_cube_tile_sizes(coff):
    """Host-side mirror of the kernel's coff-dependent tiling derivation.

    Returns (wkv_tile_m, wkv_tile_k, wkv_tile_n, dx_tile_m, dx_tile_k, dx_tile_n)
    matching the per-coff specialization in ``block_cube_kernel``.  Used only for
    host-side alignment checks (e.g. computing ``actual_num_cores``); the kernel
    re-derives the same values from the compile-time Coff TilingKey field.
    """
    if coff == 1:
        return 128, 64, 128, 128, 128, 128
    return 256, 32, 64, 64, 64, 64   # coff == 2


def _run_and_compare(inputs, coff, layout, dataType, case_name,
                     b_size=1, s_size=128, h_size=4096, d_size=128, cmp_ratio=4):
    """Run block_cube_kernel and compare against golden stage-D formula."""
    device = get_device()
    c_size = coff * d_size
    torch_dtype = torch.bfloat16 if dataType == 0 else torch.float16
    T = b_size * s_size

    key = {"Coff": coff, "Layout": layout, "DataType": dataType}
    tiling = CompressorTiling(
        batchSize=b_size,
        cmpRatio=cmp_ratio,
        tokenSize=T,
        hiddenSize=h_size,
        headDim=d_size,
        totalBlks=0,
    )

    # ── Pick the actual launch-time core count ──
    # Source: platform hardware core_num.  Clamp by alignment preconditions so
    # the K-split (d_wkv / d_wgate) and N-split (d_x) strategies stay tiled
    # without OOB loads:
    #   * K-split: per-core K = T / num_cores must be >= wkv_tile_k (single
    #     K-tile path) or a multiple of wkv_tile_k (K-accumulation path).
    #     num_cores <= T // wkv_tile_k guarantees k_per_core is a multiple of
    #     wkv_tile_k, and T % num_cores == 0 (since wkv_tile_k divides T).
    #   * N-split: per-core N = H / num_cores must be a multiple of dx_tile_n.
    #     num_cores <= H // dx_tile_n guarantees n_per_core is a multiple of
    #     dx_tile_n, and H % num_cores == 0 (since dx_tile_n divides H).
    wkv_tile_m, wkv_tile_k, wkv_tile_n, dx_tile_m, dx_tile_k, dx_tile_n = _get_cube_tile_sizes(coff)
    platform_cores = get_platform_info().core_num or 2
    actual_num_cores = min(platform_cores, T // wkv_tile_k, h_size // dx_tile_n)
    actual_num_cores = max(actual_num_cores, 1)

    # Outputs: d_wkv_out / d_wgate_out MUST be zero-init (cross-core atomic add
    # accumulates on top -- see kernel docstring).  d_x_out is 2-D (T, H) for
    # both layouts.
    d_wkv_out = torch.zeros((c_size, h_size), device=device, dtype=torch.float32)
    d_wgate_out = torch.zeros((c_size, h_size), device=device, dtype=torch.float32)
    d_x_out = torch.zeros((T, h_size), device=device, dtype=torch.float32)

    block_cube_kernel[None, actual_num_cores, key](
        inputs["x"],
        inputs["d_new_kv"], inputs["d_new_score"],
        inputs["wkv"], inputs["wgate"],
        d_x_out, d_wkv_out, d_wgate_out,
        tiling,
    )
    torch_npu.npu.synchronize()

    # ── Golden reference ──
    d_new_kv_ref = inputs["d_new_kv"].float()
    d_new_score_ref = inputs["d_new_score"].float()
    x_ref = inputs["x"].float()
    wkv_ref = inputs["wkv"].float()
    wgate_ref = inputs["wgate"].float()

    x_flat_ref = x_ref.reshape(T, h_size) if layout == 0 else x_ref

    ref_d_x = d_new_kv_ref @ wkv_ref + d_new_score_ref @ wgate_ref
    ref_d_wkv = d_new_kv_ref.t() @ x_flat_ref
    ref_d_wgate = d_new_score_ref.t() @ x_flat_ref

    # For BSH, reshape d_x_out to (B, S, H) for comparison
    if layout == 0:
        d_x_cmp = d_x_out.reshape(b_size, s_size, h_size)
        ref_d_x_cmp = ref_d_x.reshape(b_size, s_size, h_size)
    else:
        d_x_cmp = d_x_out
        ref_d_x_cmp = ref_d_x

    layout_name = "BSH" if layout == 0 else "TH"
    logging.info("[%s] %s coff=%d B=%d T=%d | d_x=%s d_wkv=%s d_wgate=%s",
                 case_name, layout_name, coff, b_size, T,
                 tuple(d_x_cmp.shape), tuple(d_wkv_out.shape), tuple(d_wgate_out.shape))

    compare(d_x_cmp.cpu(), ref_d_x_cmp.cpu(), "d_x",
            rtol=RTOL, atol=ATOL, max_error_ratio=MAX_ERROR_RATIO)
    compare(d_wkv_out.cpu(), ref_d_wkv.cpu(), "d_wkv",
            rtol=RTOL, atol=ATOL, max_error_ratio=MAX_ERROR_RATIO)
    compare(d_wgate_out.cpu(), ref_d_wgate.cpu(), "d_wgate",
            rtol=RTOL, atol=ATOL, max_error_ratio=MAX_ERROR_RATIO)
    print(f"[PRECISION_PASS] {case_name}")


def _run_case(case_name, seed, grad_scale, weight_scale, coff, layout, dataType,
              b_size=1, s_size=128, h_size=4096, d_size=128, cmp_ratio=4):
    torch_dtype = torch.bfloat16 if dataType == 0 else torch.float16
    c_size = coff * d_size
    layout_name = "BSH" if layout == 0 else "TH"
    dtype_name = "BF16" if dataType == 0 else "FP16"
    logger.info("*" * 60)
    logger.info(f"Run block_cube case: {case_name} ({layout_name}/{dtype_name}/coff={coff}/B={b_size}/D={d_size}/cr={cmp_ratio})")
    logger.info("*" * 60 + "\n")
    device = get_device()
    inputs = _gen_cube_inputs(seed, grad_scale, weight_scale, c_size, torch_dtype, layout, device,
                              b_size=b_size, s_size=s_size, h_size=h_size)
    _run_and_compare(inputs, coff, layout, dataType, case_name,
                     b_size=b_size, s_size=s_size, h_size=h_size, d_size=d_size,
                     cmp_ratio=cmp_ratio)


# ── Test cases (36 cases: full coverage of all 20 achievable path combos) ──
# Coverage dimensions (cube kernel code-path distinguishers):
#   coff {1,2}, dtype {BF16,FP16}, k_acc {F,T}, dx_m_single {T,F},
#   dx_k_single {T,F}, wkv_m_single {T,F}
# All 20 achievable (coff, dtype, k_acc, dx_m1, dx_k1, wkv_m1) combos are covered.
# cmpRatio is NOT a coverage dimension: the cube kernel does not read
# tiling.cmpRatio (only headDim/tokenSize/hiddenSize), so cr has zero effect on
# the kernel code path. cr is kept at its default value 4 for all cases.

# Each case: name -> (seed, grad_scale, weight_scale, coff, layout, dataType,
#              b_size, s_size, d_size, description)
CASES = {
    "case_01_bsh_bf16_coff2": (
        42, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 0, 0,  # coff, layout, dataType
        1, 128, 128,  # b_size, s_size, d_size
        "BSH/BF16/coff=2 base case (single-tile, no k_acc).",
    ),
    "case_02_bsh_bf16_coff1": (
        123, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 0, 0,  # coff, layout, dataType
        1, 128, 128,  # b_size, s_size, d_size
        "BSH/BF16/coff=1 base case (single M/K/N-tile).",
    ),
    "case_03_th_bf16_coff2": (
        42, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 0,  # coff, layout, dataType
        1, 128, 128,  # b_size, s_size, d_size
        "TH/BF16/coff=2 base case.",
    ),
    "case_04_th_bf16_coff1": (
        123, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 1, 0,  # coff, layout, dataType
        1, 128, 128,  # b_size, s_size, d_size
        "TH/BF16/coff=1 base case.",
    ),
    "case_05_bsh_bf16_coff2_b2": (
        42, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 0, 0,  # coff, layout, dataType
        2, 128, 128,  # b_size, s_size, d_size
        "B=2 -> T=256 (multi M-tile d_x, still no k_acc).",
    ),
    "case_06_bsh_bf16_coff2_b4": (
        0, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 0, 0,  # coff, layout, dataType
        4, 128, 128,  # b_size, s_size, d_size
        "B=4 -> T=512.",
    ),
    "case_07_th_bf16_coff2_b8": (
        7, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 0,  # coff, layout, dataType
        8, 128, 128,  # b_size, s_size, d_size
        "B=8 -> T=1024.",
    ),
    "case_08_bsh_bf16_coff1_b16": (
        123, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 0, 0,  # coff, layout, dataType
        16, 128, 128,  # b_size, s_size, d_size
        "B=16 -> T=2048 (coff=1, max B without k_acc).",
    ),
    "case_09_th_bf16_coff2_b32": (
        314, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 0,  # coff, layout, dataType
        32, 128, 128,  # b_size, s_size, d_size
        "B=32 -> T=4096 (k_acc=True, coff=2).",
    ),
    "case_10_bsh_bf16_coff2_b64": (
        1618, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 0, 0,  # coff, layout, dataType
        64, 128, 128,  # b_size, s_size, d_size
        "B=64 -> T=8192 (k_acc=True, largest batch).",
    ),
    "case_11_th_fp16_coff2_b4": (
        2718, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 1,  # coff, layout, dataType
        4, 128, 128,  # b_size, s_size, d_size
        "FP16 dtype, B=4.",
    ),
    "case_12_bsh_fp16_coff1_b8": (
        5772, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 0, 1,  # coff, layout, dataType
        8, 128, 128,  # b_size, s_size, d_size
        "FP16 dtype, coff=1, B=8.",
    ),
    "case_13_bsh_bf16_coff2_s4096": (
        42, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 0, 0,  # coff, layout, dataType
        1, 4096, 128,  # b_size, s_size, d_size
        "S=4096 -> T=4096 (k_acc=True).",
    ),
    "case_14_th_bf16_coff2_s4096": (
        0, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 0,  # coff, layout, dataType
        1, 4096, 128,  # b_size, s_size, d_size
        "S=4096, TH layout.",
    ),
    "case_15_bsh_bf16_coff1_s8192": (
        7, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 0, 0,  # coff, layout, dataType
        1, 8192, 128,  # b_size, s_size, d_size
        "S=8192, coff=1 (k_acc=True).",
    ),
    "case_16_th_bf16_coff2_s8192": (
        123, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 0,  # coff, layout, dataType
        1, 8192, 128,  # b_size, s_size, d_size
        "S=8192.",
    ),
    "case_17_bsh_bf16_coff2_s16384": (
        314, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 0, 0,  # coff, layout, dataType
        1, 16384, 128,  # b_size, s_size, d_size
        "S=16384 (largest sequence).",
    ),
    "case_18_th_fp16_coff2_s16384": (
        2718, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 1,  # coff, layout, dataType
        1, 16384, 128,  # b_size, s_size, d_size
        "S=16384, FP16.",
    ),
    "case_19_bsh_bf16_coff2_b2_s4096": (
        5772, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 0, 0,  # coff, layout, dataType
        2, 4096, 128,  # b_size, s_size, d_size
        "B=2 + S=4096 -> T=8192.",
    ),
    "case_20_th_bf16_coff1_b4_s4096": (
        1618, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 1, 0,  # coff, layout, dataType
        4, 4096, 128,  # b_size, s_size, d_size
        "B=4 + S=4096 -> T=16384, coff=1.",
    ),
    "case_21_bsh_bf16_coff2_d512": (
        42, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 0, 0,  # coff, layout, dataType
        1, 128, 512,  # b_size, s_size, d_size
        "D=512 -> C=1024 (multi M/K-tile, no k_acc).",
    ),
    "case_22_th_bf16_coff2_d512": (
        0, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 0,  # coff, layout, dataType
        1, 128, 512,  # b_size, s_size, d_size
        "D=512, TH.",
    ),
    "case_23_bsh_bf16_coff1_d512": (
        7, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 0, 0,  # coff, layout, dataType
        1, 128, 512,  # b_size, s_size, d_size
        "D=512, coff=1 -> C=512.",
    ),
    "case_24_th_bf16_coff1_d512": (
        123, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 1, 0,  # coff, layout, dataType
        1, 128, 512,  # b_size, s_size, d_size
        "D=512, coff=1, TH.",
    ),
    "case_25_bsh_fp16_coff2_d512": (
        314, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 0, 1,  # coff, layout, dataType
        1, 128, 512,  # b_size, s_size, d_size
        "D=512, FP16.",
    ),
    "case_26_th_fp16_coff2_d512": (
        2718, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 1,  # coff, layout, dataType
        1, 128, 512,  # b_size, s_size, d_size
        "D=512, FP16, TH.",
    ),
    "case_27_bsh_bf16_coff2_d512_b4": (
        5772, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 0, 0,  # coff, layout, dataType
        4, 128, 512,  # b_size, s_size, d_size
        "D=512 + B=4 -> T=512.",
    ),
    "case_28_th_bf16_coff2_d512_s4096": (
        1618, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 0,  # coff, layout, dataType
        1, 4096, 512,  # b_size, s_size, d_size
        "D=512 + S=4096 (k_acc + multi M/K-tile).",
    ),
    "case_29_bsh_fp16_coff1": (
        42, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 0, 1,  # coff, layout, dataType
        1, 128, 128,  # b_size, s_size, d_size
        "FP16 coff=1 single-tile path (B=1, S=128, D=128).",
    ),
    "case_30_th_fp16_coff1_b32": (
        123, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 1, 1,  # coff, layout, dataType
        32, 128, 128,  # b_size, s_size, d_size
        "FP16 coff=1 k_acc path (B=32 -> T=4096).",
    ),
    "case_31_th_fp16_coff1_d512": (
        2718, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 1, 1,  # coff, layout, dataType
        1, 128, 512,  # b_size, s_size, d_size
        "FP16 coff=1 D=512 single M-tile (B=1).",
    ),
    "case_32_th_fp16_coff1_d512_b4": (
        5772, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 1, 1,  # coff, layout, dataType
        4, 128, 512,  # b_size, s_size, d_size
        "FP16 coff=1 D=512 B>1 no k_acc.",
    ),
    "case_33_th_fp16_coff1_d512_b32": (
        314, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 1, 1,  # coff, layout, dataType
        32, 128, 512,  # b_size, s_size, d_size
        "FP16 coff=1 D=512 k_acc (B=32).",
    ),
    "case_34_th_bf16_coff1_d512_b4": (
        1618, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 1, 0,  # coff, layout, dataType
        4, 128, 512,  # b_size, s_size, d_size
        "BF16 coff=1 D=512 B>1 no k_acc.",
    ),
    "case_35_th_bf16_coff1_d512_b32": (
        42, 0.1, 10.0,  # seed, grad_scale, weight_scale
        1, 1, 0,  # coff, layout, dataType
        32, 128, 512,  # b_size, s_size, d_size
        "BF16 coff=1 D=512 k_acc (B=32).",
    ),
    "case_36_th_fp16_coff2_d512_b16": (
        0, 0.1, 10.0,  # seed, grad_scale, weight_scale
        2, 1, 1,  # coff, layout, dataType
        16, 128, 512,  # b_size, s_size, d_size
        "FP16 coff=2 D=512 k_acc (B=16 -> T=2048).",
    )
}


def run_all_cases():
    """Run every case in CASES (data-driven; no per-case function needed)."""
    logger.info("=" * 60)
    logger.info("PyPTO Pro compressor_grad block_cube (generalized cube kernel) test")
    logger.info("=" * 60 + "\n")
    for case_name, cfg in CASES.items():
        seed, grad_scale, weight_scale, coff, layout, dataType, \
            b_size, s_size, d_size, _desc = cfg
        _run_case(case_name, seed=seed, grad_scale=grad_scale, weight_scale=weight_scale,
                  coff=coff, layout=layout, dataType=dataType,
                  b_size=b_size, s_size=s_size, d_size=d_size)
    logger.info("All {} block_cube test cases passed!".format(len(CASES)))


def main():
    run_all_cases()


if __name__ == "__main__":
    main()
