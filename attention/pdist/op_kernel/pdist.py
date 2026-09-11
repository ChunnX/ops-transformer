# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

"""pdist kernel — pairwise p-norm distances over row pairs (upper-triangular output).

Ported from the Stage-4 verified PyPTO-Pro delivery (custom/pdist/test_pdist.py,
verifier PASS 15/15, 16/16 cases) into the ops-transformer AscendC packaging:

  * Pair-lane transposed layout: 64 row-pairs of one task block map to the 64
    lanes of one fp32 VF register (lane k <-> row j0+k).  Rows are loaded as a
    [64, M_BLK] tile, transposed to [M_BLK, 64], and consumed one row (m) per
    iteration; the anchor row x[i] is broadcast per-m with BRC_B32.
  * M-blocking with ping-pong partial-sum state: M blocks of M_BLK=128 columns
    accumulate into a resident acc tile pair ([2, 64] fp32, row 0 = partial sum,
    row 1 = NaN counter for PBranch 5), rotated with explicit indices and
    ordered by vf.mem_bar.
  * PBranch folding (TilingKey): PBranch (6 values) compiles a dedicated
    instance per p-class; every variant's inner m loop is branch-free.
  * Multi-core block-domain contiguous split: the B 64-pair task blocks are
    split contiguously per core; each core locates its first block with a
    closed-form binary search over cumsegs(i).
  * dtype instances: fp32 and fp16 specializations of this single @pl.jit
    kernel; the fp16 instance widens trans/anchor with tile-level pl.cast
    CAST_NONE after the transpose, computes entirely in fp32, and narrows the
    exit segment with pl.cast CAST_RINT.

Port adaptation (declared deviation vs the Stage-4 source, math identical):
NaN constants in PBranch 1/5 finalize are materialized with the in-repo idiom
ln(0) = -inf followed by |-inf - (-inf)| = NaN (see dense_lightning_indexer_
softmax_lse_v2.py write_neg_inf_vf for the -inf twin of this idiom), because
the CANN-tree pypto_pro serializes the vf.full(float("nan")) literal as the
libc function name ``nanf``, which bisheng rejects.
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.tilingkey import TilingKeyField

# ================================================================
#  Tile dimensions and constants
# ================================================================
TP = 64            # pairs per task block = fp32 register lane count
M_BLK = 128        # M block width, shared by fp32/fp16 instances
LANES = 64         # fp32 VL; fp16 data widened to fp32 before VF consumption

# ================================================================
#  UB(Vec) addresses (32B aligned; fp32 and fp16 instances)
# ================================================================
VA_ROWS_F32 = 0x00000   # rows_group   [64,128] FP32 32768B x2 -> 0x10000
VA_TRANS_F32 = 0x10000  # trans_group  [128,64] FP32 32768B x2 -> 0x20000
VA_ANCHOR_F32 = 0x20000  # anchor_group [1,128] FP32   512B x2 -> 0x20400
VA_ACC_F32 = 0x20400    # acc_group    [2,64]  FP32   512B x2 -> 0x20800 (resident ping-pong)
VA_OUT32_F32 = 0x20800  # out32_group  [1,64]  FP32   256B x2 -> 0x20A00

VA_ROWS16 = 0x00000     # rows16_group   [64,128] FP16 16384B x2 -> 0x08000
VA_TRANS16 = 0x08000    # trans16_group  [128,64] FP16 16384B x2 -> 0x10000
VA_TRANS32 = 0x10000    # trans32_group  [128,64] FP32 32768B x2 -> 0x20000 (widen destination)
VA_ANCHOR16 = 0x20000   # anchor16_group [1,128] FP16   256B x2 -> 0x20200
VA_ANCHOR32 = 0x20200   # anchor32_group [1,128] FP32   512B x2 -> 0x20600 (widen destination)
VA_ACC_F16 = 0x20600    # acc_group      [2,64]  FP32   512B x2 -> 0x20A00 (resident ping-pong)
VA_OUT32_F16 = 0x20A00  # out32_group    [1,64]  FP32   256B x2 -> 0x20C00
VA_OUT16 = 0x20C00      # out16_group    [1,64]  FP16   128B x2 -> 0x20D00 (exit narrowing)


# ================================================================
#  Tiling data (host tiling fills; kernel consumes)
# ================================================================
@dataclass
class PdistTiling:
    N: int        # INDEX -> int64_t (idx arithmetic safe: N=65535 -> i*(2N-i-1) ~ 8.59e9 > int32)
    M: int
    B: int        # total 64-pair task blocks (host closed form, see below)
    p_val: float  # FP32, used by PBranch 5
    inv_p: float  # FP32, used by PBranch 5


# ================================================================
#  TilingKey
# ================================================================
class PdistTilingKey:
    PBranch = TilingKeyField(bits=3, values=[0, 1, 2, 3, 4, 5])  # p branch table


# ================================================================
#  VF helper functions
# ================================================================


@pl.vector_function
def pdist_accum_init_vf(trans_tile, anchor_tile, acc_out, vc: pl.DT_INT64, p_val: pl.DT_FP32):
    """First M block of a task: acc/nan_cnt initialized from full(0.0)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    acc = vf.full(0.0, preg, dtype=pl.DT_FP32)
    if PBranch == 0:
        ones_r = vf.full(1.0, preg, dtype=pl.DT_FP32)
        zeros_r = vf.full(0.0, preg, dtype=pl.DT_FP32)
    elif PBranch == 5:
        one_r = vf.full(1.0, preg, dtype=pl.DT_FP32)
        zeros_r = vf.full(0.0, preg, dtype=pl.DT_FP32)
        nan_cnt = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, vc):
        bcast = vf.load_align(anchor_tile, m, dist=pl.LoadDist.BRC_B32)  # x[i, m0+m] -> 64 lanes
        rv = vf.load_align(trans_tile, m * LANES)                        # lane k = x[j0+k, m0+m]
        d = vf.abs_sub(bcast, rv, preg)                                  # |x_i - x_jk|
        if PBranch == 0:
            # Hamming: NaN dims count as mismatched (ne(NaN,0)=1, golden semantics)
            ne_m = vf.ne(d, 0.0, preg)
            dp = vf.select(ones_r, zeros_r, ne_m)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 1:
            # Chebyshev comparison max (GPU/CUDA aligned): NaN never wins the
            # comparison (gt(NaN, acc) is false), so NaN dims are ignored and an
            # all-NaN pair keeps acc = 0 — identical to CUDA `if (diff > agg)`.
            gt_m = vf.gt(d, acc, preg)
            acc = vf.select(d, acc, gt_m)
        elif PBranch == 2:
            dp = vf.mul(d, d, preg)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 3:
            acc = vf.add(acc, d, preg)
        elif PBranch == 4:
            dp = vf.sqrt(d, preg)  # d=|.|>=0, in domain
            acc = vf.add(acc, dp, preg)
        else:
            # PBranch 5 (general 0<p<inf): d^p = exp(p*ln d), d=0 clamped to 1 (zero contribution)
            nz = vf.ne(d, 0.0, preg)
            safe = vf.select(d, one_r, nz)
            ld = vf.ln(safe, preg)
            ld = vf.muls(ld, p_val, preg)
            ed = vf.exp(ld, preg)
            dp = vf.select(ed, zeros_r, nz)
            acc = vf.add(acc, dp, preg)
            nan_m = vf.ne(d, d, preg)
            nan_f = vf.select(one_r, zeros_r, nan_m)
            nan_cnt = vf.add(nan_cnt, nan_f, preg)
    vf.store_align(acc_out, acc, preg)
    if PBranch == 5:
        vf.store_align(acc_out + 64, nan_cnt, preg)
    # VST_VLD: this block's store -> next block/finalize load (RAW)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    # VLD_VST: this block's load -> rewrite of the same slot two blocks later (WAR)
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)


@pl.vector_function
def pdist_accum_acc_vf(trans_tile, anchor_tile, acc_in, acc_out, vc: pl.DT_INT64, p_val: pl.DT_FP32):
    """Subsequent M blocks: read the previous partial sum (acc_in = slot mblk%2) and keep
    accumulating, writing slot (mblk+1)%2."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    acc = vf.load_align(acc_in, 0)
    if PBranch == 5:
        nan_cnt = vf.load_align(acc_in, 64)
    if PBranch == 0:
        ones_r = vf.full(1.0, preg, dtype=pl.DT_FP32)
        zeros_r = vf.full(0.0, preg, dtype=pl.DT_FP32)
    elif PBranch == 5:
        one_r = vf.full(1.0, preg, dtype=pl.DT_FP32)
        zeros_r = vf.full(0.0, preg, dtype=pl.DT_FP32)
    for m in pl.range(0, vc):
        bcast = vf.load_align(anchor_tile, m, dist=pl.LoadDist.BRC_B32)  # x[i, m0+m] -> 64 lanes
        rv = vf.load_align(trans_tile, m * LANES)                        # lane k = x[j0+k, m0+m]
        d = vf.abs_sub(bcast, rv, preg)                                  # |x_i - x_jk|
        if PBranch == 0:
            ne_m = vf.ne(d, 0.0, preg)
            dp = vf.select(ones_r, zeros_r, ne_m)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 1:
            # Chebyshev comparison max (GPU/CUDA aligned): NaN never wins the
            # comparison (gt(NaN, acc) is false), so NaN dims are ignored and an
            # all-NaN pair keeps acc = 0 — identical to CUDA `if (diff > agg)`.
            gt_m = vf.gt(d, acc, preg)
            acc = vf.select(d, acc, gt_m)
        elif PBranch == 2:
            dp = vf.mul(d, d, preg)
            acc = vf.add(acc, dp, preg)
        elif PBranch == 3:
            acc = vf.add(acc, d, preg)
        elif PBranch == 4:
            dp = vf.sqrt(d, preg)
            acc = vf.add(acc, dp, preg)
        else:
            nz = vf.ne(d, 0.0, preg)
            safe = vf.select(d, one_r, nz)
            ld = vf.ln(safe, preg)
            ld = vf.muls(ld, p_val, preg)
            ed = vf.exp(ld, preg)
            dp = vf.select(ed, zeros_r, nz)
            acc = vf.add(acc, dp, preg)
            nan_m = vf.ne(d, d, preg)
            nan_f = vf.select(one_r, zeros_r, nan_m)
            nan_cnt = vf.add(nan_cnt, nan_f, preg)
    vf.store_align(acc_out, acc, preg)
    if PBranch == 5:
        vf.store_align(acc_out + 64, nan_cnt, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)


@pl.vector_function
def pdist_finalize_vf(acc_tile, out_tile, vp: pl.DT_INT64, inv_p: pl.DT_FP32):
    """Task tail: inv-pow + pair-tail mask (update_mask(vp)) + store.

    NaN constants are materialized as |-inf - (-inf)| with -inf = ln(0)
    (in-repo idiom; avoids the float("nan") literal, see module docstring).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    acc = vf.load_align(acc_tile, 0)
    if PBranch == 0:
        y = acc  # identity: count <= M < 2^24, exact in fp32
    elif PBranch == 1:
        # GPU-aligned comparison max already ignored NaN dims (all-NaN pair -> 0)
        y = acc
    elif PBranch == 2:
        y = vf.sqrt(acc, preg)
    elif PBranch == 3:
        y = acc
    elif PBranch == 4:
        y = vf.mul(acc, acc, preg)
    else:
        # PBranch 5: y = s^(1/p) = exp(ln(s)/p); s=0 -> 0; NaN count -> NaN
        nan_cnt = vf.load_align(acc_tile, 64)
        nz = vf.ne(acc, 0.0, preg)
        one_r = vf.full(1.0, preg, dtype=pl.DT_FP32)
        zero_r = vf.full(0.0, preg, dtype=pl.DT_FP32)
        s_safe = vf.select(acc, one_r, nz)  # s=0 -> clamp to 1 (ln domain guard)
        ls = vf.ln(s_safe, preg)
        ls = vf.muls(ls, inv_p, preg)
        es = vf.exp(ls, preg)
        y = vf.select(es, zero_r, nz)       # s=0 -> 0 (0^p=0, p>0)
        bad = vf.ne(nan_cnt, 0.0, preg)
        ninf_r = vf.ln(zero_r, preg)               # log(0) = -inf
        nan_r = vf.abs_sub(ninf_r, ninf_r, preg)   # NaN
        y = vf.select(nan_r, y, bad)
    mreg = vf.update_mask(vp, dtype=pl.DT_FP32)  # pair tail: only vp lanes written
    vf.store_align(out_tile, y, mreg)
    # WAR: this task's acc read -> next task's rewrite of the acc slot
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)


# ================================================================
#  Kernel
# ================================================================


@pl.jit(auto_mutex=True, tiling_key=PdistTilingKey, datatype={"x": "io_dtype", "y": "io_dtype"})
def pdist(x: pl.Ptr[pl.DT_UINT8], y: pl.Ptr[pl.DT_UINT8],
          workspace: pl.Ptr[pl.DT_UINT8], tiling: PdistTiling):
    """pdist pair-lane distance kernel (main @pl.jit kernel, named after the op file).

    x: [N, M] GM (fp32 or fp16, ND row-major); y: [N(N-1)/2] GM (same dtype as x).
    workspace: unused by this kernel, but REQUIRED by the AscendC opc ABI — the runtime
    passes (inputs..., workspace, tiling) in the args area, so the signature must keep
    workspace as the second-to-last parameter (see dense_lightning_indexer_softmax_lse_v2).
    io_dtype is the datatype-specialization compile-time constant: the
    if io_dtype == pl.DT_FP16 branch folds at parse time, so each dtype instance
    compiles only its own address table (fp32 / fp16).
    """
    N = tiling.N
    M = tiling.M
    B = tiling.B
    L = N * (N - 1) // 2
    tensor_x = pl.make_tensor(x, [N, M], dtype=io_dtype)
    tensor_y = pl.make_tensor(y, [1, L], dtype=io_dtype)  # 1-D output as a [1, L] contiguous view

    with pl.section_vector():
        core_id = pl.get_block_idx()
        # opc MIX delivery (KERNEL_TYPE_MIX_AIC_1_2): blockDim counts tsch blocks, each
        # AIV block carries get_subblock_num() (task ration) vector cores; pl.get_block_idx()
        # is the unique vector-core id (blockIdx * subblockdim + subblockid), while
        # pl.get_block_num() returns the raw AIV block count. Multiply to get the total
        # vector-core count (equals the direct-launch blockDim semantics).
        num_cores = pl.get_block_num() * pl.get_subblock_num()
        # block-domain contiguous split: core c owns [lo, hi)
        lo = core_id * B // num_cores
        hi = (core_id + 1) * B // num_cores

        # closed-form binary search: lo -> (i, jseg), fixed 17 iterations
        # cumsegs(i) = i + F(N-2) - F(N-2-i), F(x) = 32q(q-1) + qr (x+1 = 64q + r)
        qf = (N - 1) // 64
        rf = N - 1 - qf * 64
        F2 = 32 * qf * (qf - 1) + qf * rf
        i = 0
        i_hi = N - 2
        for _b in pl.range(0, 17):
            if i < i_hi:
                mid = (i + i_hi + 1) // 2
                tm = N - 1 - mid
                qm = tm // 64
                fm = 32 * qm * (qm - 1) + qm * (tm - qm * 64)
                if mid + F2 - fm <= lo:
                    i = mid
                else:
                    i_hi = mid - 1
        ti = N - 1 - i
        qi = ti // 64
        fi = 32 * qi * (qi - 1) + qi * (ti - qi * 64)
        jseg = lo - (i + F2 - fi)

        if io_dtype == pl.DT_FP16:
            # ===== fp16 instance =====
            rows16_group = pl.make_tile_group(
                type=pl.TileType(shape=[TP, M_BLK], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_ROWS16, mutex_ids=[0, 1])
            trans16_group = pl.make_tile_group(
                type=pl.TileType(shape=[M_BLK, TP], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_TRANS16, mutex_ids=[2, 3])
            anchor16_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, M_BLK], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_ANCHOR16, mutex_ids=[4, 5])
            trans32_group = pl.make_tile_group(
                type=pl.TileType(shape=[M_BLK, TP], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_TRANS32, mutex_ids=[8, 9])
            anchor32_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, M_BLK], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_ANCHOR32, mutex_ids=[10, 11])
            acc_group = pl.make_tile_group(
                type=pl.TileType(shape=[2, TP], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=VA_ACC_F16, depth=2)  # pure-V resident ping-pong: no mutex
            out32_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, TP], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_OUT32_F16, mutex_ids=[6, 7])
            out16_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, TP], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_OUT16, mutex_ids=[12, 13])

            for t in pl.range(0, hi - lo):
                j0 = i + 1 + jseg * TP
                vp = pl.min(TP, N - 1 - i - jseg * TP)     # pair tail
                base = (i * (2 * N - i - 1)) // 2 + jseg * TP  # int64 upper-triangular base
                n_mblk = (M + M_BLK - 1) // M_BLK
                for mblk in pl.range(0, n_mblk):
                    m0 = mblk * M_BLK
                    vc = pl.min(M_BLK, M - m0)             # M tail
                    rows_slot = rows16_group.next()
                    pl.set_validshape(rows_slot, [vp, vc])
                    anchor_slot = anchor16_group.next()
                    pl.set_validshape(anchor_slot, [1, vc])
                    trans16_slot = trans16_group.next()
                    pl.set_validshape(trans16_slot, [vc, vp])
                    trans32_slot = trans32_group.next()
                    pl.set_validshape(trans32_slot, [vc, vp])
                    anchor32_slot = anchor32_group.next()
                    pl.set_validshape(anchor32_slot, [1, vc])
                    pl.load(rows_slot, tensor_x, [j0, m0])
                    pl.load(anchor_slot, tensor_x, [i, m0])
                    pl.transpose(trans16_slot, rows_slot)   # V pipe: [64,128] -> [128,64]
                    # widen after the transpose, before VF consumption
                    pl.cast(trans32_slot, trans16_slot, mode=pl.RoundMode.CAST_NONE)
                    pl.cast(anchor32_slot, anchor_slot, mode=pl.RoundMode.CAST_NONE)
                    if mblk == 0:
                        pdist_accum_init_vf(trans32_slot, anchor32_slot,
                                            acc_group[1], vc, tiling.p_val)
                    else:
                        pdist_accum_acc_vf(trans32_slot, anchor32_slot,
                                           acc_group[mblk % 2], acc_group[(mblk + 1) % 2],
                                           vc, tiling.p_val)
                out32_slot = out32_group.next()
                pl.set_validshape(out32_slot, [1, vp])
                pdist_finalize_vf(acc_group[n_mblk % 2], out32_slot, vp, tiling.inv_p)
                out16_slot = out16_group.next()
                pl.set_validshape(out16_slot, [1, vp])
                pl.cast(out16_slot, out32_slot, mode=pl.RoundMode.CAST_RINT)  # exit narrowing
                pl.store(tensor_y, out16_slot, [0, base])   # MTE3 overwrite
                jseg = jseg + 1                            # sequential advance (<= 1 row change per step)
                segs_i = (N - 2 - i) // TP + 1
                if jseg == segs_i:
                    i = i + 1
                    jseg = 0
        else:
            # ===== fp32 instance =====
            rows_group = pl.make_tile_group(
                type=pl.TileType(shape=[TP, M_BLK], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_ROWS_F32, mutex_ids=[0, 1])
            trans_group = pl.make_tile_group(
                type=pl.TileType(shape=[M_BLK, TP], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_TRANS_F32, mutex_ids=[2, 3])
            anchor_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, M_BLK], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_ANCHOR_F32, mutex_ids=[4, 5])
            acc_group = pl.make_tile_group(
                type=pl.TileType(shape=[2, TP], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec),
                addrs=VA_ACC_F32, depth=2)  # pure-V resident ping-pong: no mutex
            out32_group = pl.make_tile_group(
                type=pl.TileType(shape=[1, TP], dtype=pl.DT_FP32,
                                 target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]),
                addrs=VA_OUT32_F32, mutex_ids=[6, 7])

            for t in pl.range(0, hi - lo):
                j0 = i + 1 + jseg * TP
                vp = pl.min(TP, N - 1 - i - jseg * TP)
                base = (i * (2 * N - i - 1)) // 2 + jseg * TP
                n_mblk = (M + M_BLK - 1) // M_BLK
                for mblk in pl.range(0, n_mblk):
                    m0 = mblk * M_BLK
                    vc = pl.min(M_BLK, M - m0)
                    rows_slot = rows_group.next()
                    pl.set_validshape(rows_slot, [vp, vc])
                    anchor_slot = anchor_group.next()
                    pl.set_validshape(anchor_slot, [1, vc])
                    trans_slot = trans_group.next()
                    pl.set_validshape(trans_slot, [vc, vp])
                    pl.load(rows_slot, tensor_x, [j0, m0])
                    pl.load(anchor_slot, tensor_x, [i, m0])
                    pl.transpose(trans_slot, rows_slot)
                    if mblk == 0:
                        pdist_accum_init_vf(trans_slot, anchor_slot,
                                            acc_group[1], vc, tiling.p_val)
                    else:
                        pdist_accum_acc_vf(trans_slot, anchor_slot,
                                           acc_group[mblk % 2], acc_group[(mblk + 1) % 2],
                                           vc, tiling.p_val)
                out_slot = out32_group.next()
                pl.set_validshape(out_slot, [1, vp])
                pdist_finalize_vf(acc_group[n_mblk % 2], out_slot, vp, tiling.inv_p)
                pl.store(tensor_y, out_slot, [0, base])
                jseg = jseg + 1
                segs_i = (N - 2 - i) // TP + 1
                if jseg == segs_i:
                    i = i + 1
                    jseg = 0
