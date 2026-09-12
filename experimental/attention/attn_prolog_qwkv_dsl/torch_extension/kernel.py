# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fused ``attn_prolog_qwkv`` NPU kernel (CANNBotDSL, Ascend950 / dav-3510).

One MIX_AIC_1_2 kernel, two cube paths and one vector epilogue::

    C_w  BF16  x @ ww^T  --fixpipe(deq_scale=softmax_scale)-->  GM w
    C_q  MXFP8 qr @ wqb^T  =fixpipe(dual_dst)=>  cv_ub  --VF-->  GM q, descale_q

Tiling for the frozen case ``dim=5120, q_lora=1280, N=32, D=128, Dr=64``: the
``T`` axis is split over AICs in tiles of 128 rows, ``qr`` plus its MX scale
stay resident in L1 for the whole 32-head loop, ``wqb`` streams in 80 KB K
windows, and the W path's ``x`` traffic is chopped into 32 windows retired one
per head so it never stalls the cube.  The cube roofline at T=4096 is about 52.7 us vs HBM 38.7 us.

Everything the vector unit does is raw-VF register code.  ``vcast`` has no fp4
lowering in this wheel (verified), so E2M1 nibbles are produced by integer bit
manipulation and a small gather table, then packed two-per-byte.
"""

from __future__ import annotations

import os

import torch

from cannbotdsl import (
    Buffer,
    Channel,
    ChannelKind,
    Float8E4M3FN,
    Float8E8M0,
    MemLoc,
    Tensor,
    const_expr,
    dtypes,
    get_block_idx,
    get_block_num,
    get_subblock_id,
    jit,
    kernel,
    matmul,
    range_constexpr,
)
from cannbotdsl.raw_reg import (
    PackMode,
    create_mask,
    full_mask,
    varange,
    vabs,
    vadd,
    vadds,
    vbitwise_and,
    vbitwise_or,
    vdups,
    vgather,
    vgather_reg,
    vges,
    vgts,
    vload,
    vlts,
    vmadd,
    vmaxs,
    vmul,
    vne,
    vcast,
    vmem_bar,
    vpair_reduce_sum,
    vreduce_max,
    vreinterpret,
    vselect,
    vshl,
    vshr,
    vstore,
    vstore_first,
    vstore_pack,
    vsub,
)
from cannbotdsl.tensor import (
    local_slice,
    make_copy_engine,
    make_partition_tiler,
    mem_copy,
    partition_view,
    tile_view,
)
from cannbotdsl.vf import vf

__all__ = ["AttnPrologQwkv", "build_e2m1_lut", "plan"]

I32 = dtypes.int32
U32 = dtypes.uint32
F32 = dtypes.float32
U8 = dtypes.uint8

VL = 64  # fp32 lanes in one 256-byte raw vector register
MX_GROUP = 32  # elements per E8M0 scale
MX_K_ALIGN = 64  # K elements per public paired-scale group
MX_PAIR = 2
E8M0_BIAS = 127
FP4_E2M1_MAX = 6.0
FP32_EXP_SHIFT = 23
FP32_MANTISSA_MASK = 0x7FFFFF
FP32_HALF_MANTISSA = 0x400000  # significand exactly 1.5
FP32_MAG_MASK = 0x7FFFFFFF
# |y| <= 6 always, so (bits(|y|) >> LUT_SHIFT) <= 1036.
LUT_SHIFT = 20
LUT_LEN = 1088

CUBE_CORES = 32

# Channel depths, overridable while tuning so a sweep is one env var per run.
# Read at trace time, so every one of these is a compile-time constant.
def _depth(name: str, default: int) -> int:
    return int(os.environ.get(f"APQ_{name}_DEPTH", default))


D_CV = _depth("CV", 2)
D_QUB = _depth("QUB", 2)
D_WQB = _depth("WQB", 2)
D_L0C_Q = _depth("L0C_Q", 2)
D_L0AB_Q = _depth("L0AB_Q", 2)
D_L0B_Q = _depth("L0B_Q", D_L0AB_Q)
D_X = _depth("X", 2)
# The W path's L0 staging is deliberately single-buffered.  Every L0A byte it
# gives up buys a larger ``base_k_w`` (32 -> 80 for the frozen case), and a
# fatter, rarer mmad is worth far more here than double buffering: W is 3% of
# the cube work but its mmad count was setting the rate at which the Q path
# could get its own L0 loads through the in-order MTE1 pipe.  Measured at
# T=4096: 80.3 us -> 70.2 us, chip MFU 65.7% -> 75.2%.
D_L0AB_W = _depth("L0AB_W", 1)

# W-path schedule inside the Q head loop.  MTE2 is one in-order pipe, so where
# the W fills sit relative to the Q fills decides whether Q's mmads wait behind
# them.  "wfirst" is the original: W fills and W mmads both ahead of Q.
W_SCHED = os.environ.get("APQ_W_SCHED", "wfirst")

# Diagnostic-only ablations; some produce wrong numbers by construction and
# exist purely to attribute wall time to a pipe.  "" is the real kernel.
PROBE = os.environ.get("APQ_PROBE", "")

# L2 retention hints.  ``wqb`` is re-read by every core and wants to stay in L2;
# ``x`` is streamed once per row and does not.
L2_Q = int(os.environ.get("APQ_L2_Q", 1))
L2_W = int(os.environ.get("APQ_L2_W", 1))
UNIT_FLAG = int(os.environ.get("APQ_UNIT_FLAG", 0))
# Split the head axis over leftover AICs when T-tiles alone cannot fill the
# chip.  T>=4096 (32 T-tiles) keeps the original T-only schedule so the 75%
# MFU path is unchanged.  APQ_HEAD_SPLIT=0 restores the old 1-core-per-T-tile
# launch for A/B.
HEAD_SPLIT = int(os.environ.get("APQ_HEAD_SPLIT", 1))

L1_BYTES = 512 * 1024
L0AB_BYTES = 64 * 1024
L0C_BYTES = 256 * 1024
UB_BYTES = 248 * 1024


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def scale_l1_len(k: int) -> int:
    """Contiguous E8M0 length in L1 for a K extent (paired public layout)."""
    return ceil_div(k, MX_K_ALIGN) * MX_PAIR


# ---------------------------------------------------------------------------
# E2M1 rounding table
# ---------------------------------------------------------------------------

def build_e2m1_lut() -> torch.Tensor:
    """Gather table mapping ``bits(|y|) >> 20`` to a 3-bit E2M1 magnitude code.

    ``y`` is the already-descaled value, so ``|y| <= 6``.  Index bits are the
    biased fp32 exponent times eight plus the top three mantissa bits; every
    E2M1 midpoint (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0) is exactly on such a
    bucket boundary, so a truncating index reproduces round-to-nearest with
    ties away from zero -- the rule the CPU golden uses.
    """
    codes = torch.zeros(LUT_LEN, dtype=torch.int32)
    midpoints = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    for exponent in range(120, 133):
        for mantissa in range(8):
            index = exponent * 8 + mantissa
            if index >= LUT_LEN:
                continue
            value = 2.0 ** (exponent - E8M0_BIAS) * (1.0 + mantissa / 8.0)
            codes[index] = sum(1 for m in midpoints if value >= m)
    codes[133 * 8:] = 7
    return codes


# ---------------------------------------------------------------------------
# Host tiling
# ---------------------------------------------------------------------------

class Plan:
    """Static tile plan; every field is a compile-time constant."""

    def __init__(self, t: int, dim: int, q_lora: int, n_heads: int, d: int, dr: int):
        self.t = t
        self.dim = dim
        self.q_lora = q_lora
        self.n_heads = n_heads
        self.d = d
        self.dr = dr

        self.base_m = 128 if t >= 128 else max(16, ceil_div(t, 16) * 16)
        self.rows_vec = self.base_m // 2
        self.n_tiles = ceil_div(t, self.base_m)
        self._plan_head_split()

        # W path always produces the full (T_tile, N) tile: splitting it to
        # N=1 per core turns the BF16 GEMM into a GEMV and, worse, makes
        # every AIC read the whole ``x`` tile from HBM.  When heads are
        # split, only the h_block==0 core of each T-tile runs W
        # (``w_leader_only``); the others skip it so ``x`` is read once.
        self.base_n_w = n_heads
        self.w_leader_only = self.n_head_blocks > 1
        self.k_l1_w = self._pick_k_l1_w()
        self.base_k_w = self._pick_base_k_w()
        self.step_k_w = self.k_l1_w // self.base_k_w
        self.k_l1_tiles_w = ceil_div(dim, self.k_l1_w)
        self.w_interleaved = self.k_l1_tiles_w % self.heads_per_block == 0
        # Interleaving W under Q only pays when this core has enough Q heads
        # to hide the ``x`` traffic.  A 1-head leader cannot hide 1.25 MB, so
        # it runs W as one contiguous reduction instead.
        self.w_separate = self.w_leader_only and self.heads_per_block <= 8
        self.w_per_head = (
            0
            if self.w_separate
            else (self.k_l1_tiles_w // self.heads_per_block if self.w_interleaved else 0)
        )

        # Q path: one head per output tile keeps the full D on chip for RoPE.
        self.base_n_q = d
        self.base_k_q = int(os.environ.get("APQ_BASE_K_Q", 0)) or min(
            128, ceil_div(q_lora, MX_K_ALIGN) * MX_K_ALIGN
        )
        self.k_l1_q = self._pick_k_l1_q()
        self.step_k_q = self.k_l1_q // self.base_k_q
        self.k_l1_tiles_q = ceil_div(q_lora, self.k_l1_q)
        self.k_l0_tiles_q = ceil_div(q_lora, self.base_k_q)
        self.scale_k_l0_len = self.base_k_q // MX_GROUP
        self.scale_k_l1_len_q = scale_l1_len(self.k_l1_q)
        self.scale_k_len_q = scale_l1_len(q_lora)
        self.qr_k_l1 = ceil_div(q_lora, MX_K_ALIGN) * MX_K_ALIGN

        # Vector epilogue geometry.
        self.n_reg = d // VL
        self.groups = d // MX_GROUP
        self.rope_regs = dr // VL
        self.enc_regs = ceil_div(self.rows_vec * self.groups, VL)
        self.amax_len = self.enc_regs * VL

        self._check()

    def _plan_head_split(self) -> None:
        """Fill leftover AICs by splitting heads when T-tiles < 32.

        Official small-T cases (128/200/256) used to launch 1-2 cores and
        report 2-5% chip MFU; the missing work is the other 30 heads, which
        are independent GEMMs.  Once those heads occupy the chip the shape
        is HBM-bound (``wqb`` is 5.24 MB regardless of T), so the target
        flips from MFU to MBU.
        """
        if not HEAD_SPLIT or self.n_tiles >= CUBE_CORES:
            self.n_head_blocks = 1
            self.heads_per_block = self.n_heads
            self.used_cores = min(CUBE_CORES, self.n_tiles)
            return
        n_hb = min(self.n_heads, CUBE_CORES // max(self.n_tiles, 1))
        while n_hb > 1 and self.n_heads % n_hb:
            n_hb -= 1
        self.n_head_blocks = max(n_hb, 1)
        self.heads_per_block = self.n_heads // self.n_head_blocks
        self.used_cores = min(CUBE_CORES, self.n_tiles * self.n_head_blocks)

    def _pick_k_l1_w(self) -> int:
        """Largest <=256 K window dividing dim whose window count divides heads_per_block.

        dim=5120, heads_per_block=32 gives 160: exactly one window per Q head.
        With a 1-head block the same search picks 256, fewer, fatter bursts.
        """
        forced = int(os.environ.get("APQ_K_L1_W", 0))
        if forced:
            return forced
        best = 0
        for k in range(32, 257, 32):
            if self.dim % k:
                continue
            if (self.dim // k) % self.heads_per_block == 0:
                best = k
        return best if best else 128

    def _pick_base_k_w(self) -> int:
        """L0 K step for the W path: as large as L0A allows next to the Q path."""
        q_l0a = D_L0AB_Q * self.base_m * (
            int(os.environ.get("APQ_BASE_K_Q", 0))
            or min(128, ceil_div(self.q_lora, MX_K_ALIGN) * MX_K_ALIGN)
        )
        budget = L0AB_BYTES - q_l0a
        best = 16
        for k in range(16, self.k_l1_w + 1, 16):
            if self.k_l1_w % k:
                continue
            if D_L0AB_W * self.base_m * k * 2 <= budget:
                best = k
        return best

    def row_split(self, rows, cols: int):
        """Row-balanced cube-tile -> vector-subblock tiler for a ``cols``-wide view."""
        return make_partition_tiler((rows, cols), (self.base_m, cols))

    def _pick_k_l1_q(self) -> int:
        """Largest K window that keeps double-buffered wqb plus resident qr in L1."""
        resident = self.base_m * ceil_div(self.q_lora, MX_K_ALIGN) * MX_K_ALIGN
        resident += self.base_m * scale_l1_len(self.q_lora)
        # Leave room for the W path staging buffers.
        w_bytes = D_X * (self.base_m + self.base_n_w) * self.k_l1_w * 2
        budget = L1_BYTES - resident - w_bytes
        best = self.base_k_q
        step = 1
        while step * self.base_k_q <= self.q_lora:
            k_l1 = step * self.base_k_q
            if k_l1 % MX_K_ALIGN:
                step += 1
                continue
            need = D_WQB * self.base_n_q * (k_l1 + scale_l1_len(k_l1))
            if need > budget:
                break
            if self.q_lora % k_l1 == 0:
                best = k_l1
            step += 1
        return best

    def _check(self) -> None:
        if self.d % VL or self.dr % VL:
            raise NotImplementedError(
                f"fused kernel needs D and Dr as multiples of {VL}, got D={self.d}, Dr={self.dr}"
            )
        if self.rope_regs != 1:
            # Dr == 64 keeps rotate-half inside one register, so pass 1 can read
            # the pre-RoPE partner lanes from the register it is about to
            # overwrite.  Dr > 64 would need every rope register live at once
            # before any write-back; not implemented, and Dr=64 is the frozen
            # case (and what MLA uses).
            raise NotImplementedError(
                f"rotate-half is implemented for Dr == {VL} only, got Dr={self.dr}"
            )
        if self.dr > self.d:
            raise ValueError(f"Dr={self.dr} exceeds D={self.d}")
        if self.q_lora % MX_K_ALIGN:
            raise NotImplementedError(
                f"q_lora must be a multiple of {MX_K_ALIGN}, got {self.q_lora}"
            )
        if self.dim % self.k_l1_w:
            raise NotImplementedError(
                f"dim must be a multiple of {self.k_l1_w}, got {self.dim}"
            )
        self.assert_budgets()

    def assert_budgets(self) -> None:
        l1 = (
            self.base_m * self.qr_k_l1  # qr resident
            + self.base_m * self.scale_k_len_q
            + D_WQB * self.base_n_q * self.k_l1_q  # wqb, multi-buffered
            + D_WQB * self.scale_k_l1_len_q * self.base_n_q
            + D_X * self.base_m * self.k_l1_w * 2  # x
            + D_X * self.base_n_w * self.k_l1_w * 2  # ww
        )
        l0a = (
            D_L0AB_Q * self.base_m * self.base_k_q
            + D_L0AB_W * self.base_m * self.base_k_w * 2
        )
        l0b = (
            D_L0B_Q * self.base_n_q * self.base_k_q
            + D_L0AB_W * self.base_n_w * self.base_k_w * 2
        )
        l0c = 4 * (
            D_L0C_Q * self.base_m * self.base_n_q + 2 * self.base_m * self.base_n_w
        )
        ub = (
            D_CV * self.rows_vec * self.d * 4  # cv_ub
            + D_QUB * self.rows_vec * (self.d // 2)  # q_ub
            + D_QUB * self.rows_vec * self.groups  # descale
            + self.amax_len * 8  # amax fp32 + off int32
            + LUT_LEN * 4
            + 2 * self.rows_vec * max(self.dr, 1) * 4  # cos + sin
        )
        for name, used, cap in (
            ("L1", l1, L1_BYTES),
            ("L0A", l0a, L0AB_BYTES),
            ("L0B", l0b, L0AB_BYTES),
            ("L0C", l0c, L0C_BYTES),
            ("UB", ub, UB_BYTES),
        ):
            if used > cap:
                raise ValueError(
                    f"{name} budget exceeded: {used} > {cap} bytes; retune Plan"
                )
        self.bytes_used = {"L1": l1, "L0A": l0a, "L0B": l0b, "L0C": l0c, "UB": ub}


def plan(t: int, dim: int, q_lora: int, n_heads: int, d: int, dr: int) -> Plan:
    return Plan(t, dim, q_lora, n_heads, d, dr)


# ---------------------------------------------------------------------------
# Cube modules
# ---------------------------------------------------------------------------

class CubeQ:
    """MXFP8 ``qr @ wqb^T`` for one (T tile, head); owns its L1/L0 staging."""

    def __init__(self, p: Plan):
        self.p = p
        self.qr_l1 = Channel(
            MemLoc.L1, (p.base_m, p.qr_k_l1), Float8E4M3FN, depth=1, data_format="nz"
        )
        self.scale_qr_l1 = Channel(
            MemLoc.L1, (p.base_m, p.scale_k_len_q), Float8E8M0, depth=1, data_format="zn"
        )
        self.wqb_l1 = Channel(
            MemLoc.L1, (p.base_n_q, p.k_l1_q), Float8E4M3FN, depth=D_WQB, data_format="nz"
        )
        self.scale_wqb_l1 = Channel(
            MemLoc.L1,
            (p.scale_k_l1_len_q, p.base_n_q),
            Float8E8M0,
            depth=D_WQB,
            data_format="nz",
        )
        self.l0a = Channel(MemLoc.L0A, (p.base_m, p.base_k_q), Float8E4M3FN, depth=D_L0AB_Q)
        self.l0b = Channel(MemLoc.L0B, (p.base_n_q, p.base_k_q), Float8E4M3FN, depth=D_L0B_Q)
        self.l0c = Channel(MemLoc.L0C, (p.base_m, p.base_n_q), F32, depth=D_L0C_Q)

        self.eng_nd2nz = make_copy_engine(
            format_transform="nd2nz", dtype=Float8E4M3FN, pad_value=0.0
        )
        self.eng_scale_a = make_copy_engine(
            format_transform="mx_scale_and", dtype=Float8E8M0, pad_value=0.0
        )
        self.eng_scale_b = make_copy_engine(
            format_transform="mx_scale_bdn", dtype=Float8E8M0, pad_value=0.0
        )
        # split-M L0C -> UB handoff requires dual_dst_ctl=1 (API note #16).
        if UNIT_FLAG:
            # Lets the fixpipe drain start on the last accumulating mmad instead
            # of after a full L0C sync, which shortens the cube -> vector chain.
            self.eng_c2v = make_copy_engine(
                dtype=F32, dual_dst_ctl=1, unit_flag_mode=3
            )
        else:
            self.eng_c2v = make_copy_engine(dtype=F32, dual_dst_ctl=1)

    @jit
    def load_qr(self, qr_gm, descale_qr_gm, tile):
        p = self.p
        mem_copy(
            self.qr_l1,
            tile_view(qr_gm, (p.base_m, p.qr_k_l1), (tile, 0)),
            engine=self.eng_nd2nz,
            l2_cache_ctl=L2_Q,
        )
        mem_copy(
            self.scale_qr_l1,
            tile_view(
                descale_qr_gm,
                (p.base_m, p.scale_k_len_q // MX_PAIR, MX_PAIR),
                (tile, 0, 0),
            ),
            engine=self.eng_scale_a,
            l2_cache_ctl=L2_Q,
        )

    @jit
    def gemm_head(self, wqb_gm, descale_wqb_gm, head):
        """Accumulate the whole q_lora reduction for one head into L0C."""
        p = self.p
        groups = p.scale_k_l1_len_q // MX_PAIR
        # Unrolled: keeps ``init`` a compile-time bool, so the hottest loop in
        # the kernel has no scf.if around its mmad.
        for kw in range_constexpr(p.k_l1_tiles_q):
            mem_copy(
                self.wqb_l1,
                tile_view(wqb_gm, (p.base_n_q, p.k_l1_q), (head, kw)),
                engine=self.eng_nd2nz,
                l2_cache_ctl=L2_Q,
            )
            mem_copy(
                self.scale_wqb_l1,
                tile_view(descale_wqb_gm, (p.base_n_q, groups, MX_PAIR), (head, kw, 0)),
                engine=self.eng_scale_b,
                l2_cache_ctl=L2_Q,
            )
            for kk in range_constexpr(p.step_k_q):
                k_l0 = kw * p.step_k_q + kk
                mem_copy(
                    self.l0a,
                    tile_view(self.qr_l1, (p.base_m, p.base_k_q), (0, k_l0)),
                    mx_scale=tile_view(
                        self.scale_qr_l1, (p.base_m, p.scale_k_l0_len), (0, k_l0)
                    ),
                )
                mem_copy(
                    self.l0b,
                    tile_view(self.wqb_l1, (p.base_n_q, p.base_k_q), (0, kk)),
                    mx_scale=tile_view(
                        self.scale_wqb_l1, (p.scale_k_l0_len, p.base_n_q), (kk, 0)
                    ),
                )
                if const_expr(UNIT_FLAG):
                    last_k = k_l0 == p.k_l0_tiles_q - 1
                    matmul(
                        self.l0c,
                        self.l0a,
                        self.l0b,
                        init=(k_l0 == 0),
                        unit_flag=3 if last_k else 2,
                    )
                else:
                    matmul(self.l0c, self.l0a, self.l0b, init=(k_l0 == 0))

    @jit
    def drain_to_vec(self, cv_ub, split_m):
        mem_copy(cv_ub, self.l0c, engine=self.eng_c2v, partition=split_m)


class CubeW:
    """BF16 ``x @ ww^T`` scaled in fixpipe; writes GM directly, no vector hop."""

    def __init__(self, p: Plan):
        self.p = p
        bf16 = dtypes.bfloat16
        self.x_l1 = Channel(
            MemLoc.L1, (p.base_m, p.k_l1_w), bf16, depth=D_X, data_format="nz"
        )
        self.ww_l1 = Channel(
            MemLoc.L1, (p.base_n_w, p.k_l1_w), bf16, depth=D_X, data_format="nz"
        )
        self.l0a = Channel(MemLoc.L0A, (p.base_m, p.base_k_w), bf16, depth=D_L0AB_W)
        self.l0b = Channel(MemLoc.L0B, (p.base_n_w, p.base_k_w), bf16, depth=D_L0AB_W)
        self.l0c = Channel(MemLoc.L0C, (p.base_m, p.base_n_w), F32, depth=2)
        self.eng_nd2nz = make_copy_engine(
            format_transform="nd2nz", dtype=bf16, pad_value=0.0
        )
        self.eng_fp = make_copy_engine(dtype=F32)

    @jit
    def gemm_head_slice(self, x_gm, ww_gm, tile, local_head, h_block, l0c_slot):
        """Retire this local head's share of the K reduction into ``l0c_slot``.

        Called from inside the Q head loop so the ``x`` traffic trickles in on
        MTE2 underneath the Q path's cube work.  ``local_head`` is 0 ..
        heads_per_block-1 on this core; ``h_block`` selects which slice of
        ``ww`` / ``w`` this core owns.  ``local_head`` is a runtime index,
        hence the manual L0C transaction: with an implicit channel the
        ``scf.if`` around the ``init`` matmul reads as a second writer.
        """
        p = self.p
        self._fill_window(
            x_gm, ww_gm, tile, h_block, local_head * p.w_per_head
        )
        for j in range_constexpr(p.w_per_head):
            if const_expr(j + 1 < p.w_per_head):
                self._fill_window(
                    x_gm, ww_gm, tile, h_block, local_head * p.w_per_head + j + 1
                )
            x_ready = self.x_l1.wait()
            ww_ready = self.ww_l1.wait()
            for kk in range_constexpr(p.step_k_w):
                mem_copy(self.l0a, tile_view(x_ready, (p.base_m, p.base_k_w), (0, kk)))
                mem_copy(self.l0b, tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk)))
                first = kk == 0 and j == 0
                if const_expr(PROBE == "no_runtime_init"):
                    matmul(l0c_slot, self.l0a, self.l0b, init=False)
                elif const_expr(PROBE == "no_w_mmad"):
                    pass
                else:
                    matmul(
                        l0c_slot,
                        self.l0a,
                        self.l0b,
                        init=(local_head == 0) if const_expr(first) else False,
                    )
            self.x_l1.release(x_ready)
            self.ww_l1.release(ww_ready)

    @jit
    def _fill_window(self, x_gm, ww_gm, tile, h_block, kw):
        p = self.p
        x_slot = self.x_l1.acquire()
        mem_copy(
            x_slot,
            tile_view(x_gm, (p.base_m, p.k_l1_w), (tile, kw)),
            engine=self.eng_nd2nz,
            l2_cache_ctl=L2_W,
        )
        self.x_l1.commit(x_slot)
        ww_slot = self.ww_l1.acquire()
        mem_copy(
            ww_slot,
                tile_view(ww_gm, (p.base_n_w, p.k_l1_w), (0, kw)),
            engine=self.eng_nd2nz,
            l2_cache_ctl=L2_W,
        )
        self.ww_l1.commit(ww_slot)

    @jit
    def fill(self, x_gm, ww_gm, tile, local_head, h_block):
        """Issue this local head's W-path L1 fills; does not wait for them."""
        p = self.p
        for j in range_constexpr(p.w_per_head):
            self._fill_window(
                x_gm, ww_gm, tile, h_block, local_head * p.w_per_head + j
            )

    @jit
    def drain(self, local_head, l0c_slot):
        """Retire the oldest staged window pair into ``l0c_slot``."""
        p = self.p
        for j in range_constexpr(p.w_per_head):
            x_ready = self.x_l1.wait()
            ww_ready = self.ww_l1.wait()
            for kk in range_constexpr(p.step_k_w):
                mem_copy(self.l0a, tile_view(x_ready, (p.base_m, p.base_k_w), (0, kk)))
                mem_copy(self.l0b, tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk)))
                first = kk == 0 and j == 0
                matmul(
                    l0c_slot,
                    self.l0a,
                    self.l0b,
                    init=(local_head == 0) if const_expr(first) else False,
                )
            self.x_l1.release(x_ready)
            self.ww_l1.release(ww_ready)

    @jit
    def gemm_full(self, x_gm, ww_gm, tile, l0c_slot):
        """All K windows of W as one reduction; used when Q cannot hide them."""
        p = self.p
        self._fill_window(x_gm, ww_gm, tile, 0, 0)
        for kw in range_constexpr(p.k_l1_tiles_w):
            if const_expr(kw + 1 < p.k_l1_tiles_w):
                self._fill_window(x_gm, ww_gm, tile, 0, kw + 1)
            x_ready = self.x_l1.wait()
            ww_ready = self.ww_l1.wait()
            for kk in range_constexpr(p.step_k_w):
                mem_copy(self.l0a, tile_view(x_ready, (p.base_m, p.base_k_w), (0, kk)))
                mem_copy(self.l0b, tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk)))
                first = kw == 0 and kk == 0
                matmul(l0c_slot, self.l0a, self.l0b, init=const_expr(first))
            self.x_l1.release(x_ready)
            self.ww_l1.release(ww_ready)

    @jit
    def store(self, w_gm, tile, h_block, softmax_scale, l0c_slot):
        # softmax_scale folded into the fixpipe DEQSCALE register: the W path
        # never touches the vector unit.
        p = self.p
        self.l0c.commit(l0c_slot)
        ready = self.l0c.wait()
        mem_copy(
            tile_view(w_gm, (p.base_m, p.base_n_w), (tile, 0)),
            ready,
            engine=self.eng_fp,
            deq_scale_val=softmax_scale,
            l2_cache_ctl=L2_W,
        )
        self.l0c.release(ready)


# ---------------------------------------------------------------------------
# Vector module
# ---------------------------------------------------------------------------

class VectorQ:
    """Inplace RoPE plus MXFP4 quantisation of one head tile, raw VF only."""

    def __init__(self, p: Plan):
        self.p = p
        rows = p.rows_vec
        # D is a multiple of 64, so q_ub's rows are already 32-byte aligned and
        # its declared stride is its real stride.  dsc_ub's natural shape
        # (rows, groups) is not: UB rounds the innermost axis up to 32 bytes, so
        # a 4-byte row would really be 32 bytes apart and the linear offsets
        # pass 2 writes at would land in the padding.  Declare it in the shape
        # pass 2 actually writes (one 64-byte register per row) and let ``store``
        # re-view it as (rows, groups) with an explicit packed stride.
        self.q_ub = Channel(MemLoc.UB, (rows, p.d // 2), U8, depth=D_QUB)
        self.dsc_ub = Channel(MemLoc.UB, (p.enc_regs, VL), U8, depth=D_QUB)
        # Buffer vs Channel is a synchronisation decision, not a sizing one.
        # amax/off are written and read only by this vector core, so a plain
        # Buffer plus vmem_bar orders them.  The LUT and the cos/sin tables
        # arrive by MTE2 DMA and are then read by VF: that crosses pipes, and a
        # Buffer carries no dependency, so the VF read can beat the DMA.  As
        # Channels (depth=1, one fill per T-tile) the handoff is tracked.
        self.amax_ub = Buffer(MemLoc.UB, (p.amax_len,), F32)
        self.off_ub = Buffer(MemLoc.UB, (p.amax_len,), I32)
        self.lut_ub = Channel(MemLoc.UB, (LUT_LEN,), I32, depth=1)
        self.cos_ub = Channel(MemLoc.UB, (rows, max(p.dr, 1)), F32, depth=1)
        self.sin_ub = Channel(MemLoc.UB, (rows, max(p.dr, 1)), F32, depth=1)

    @jit
    def load_tile_constants(self, lut_gm, cos_half, sin_half):
        """Per-T-tile prologue: DMA the LUT and the cos/sin tables into UB."""
        mem_copy(self.lut_ub, tile_view(lut_gm, (LUT_LEN,), (0,)))
        if const_expr(self.p.dr > 0):
            mem_copy(self.cos_ub, cos_half)
            mem_copy(self.sin_ub, sin_half)

    @jit
    def epilogue(self, cv_ub, rows):
        """RoPE then MXFP4 quantise ``cv_ub`` into ``q_ub`` / ``dsc_ub``."""
        p = self.p
        with vf(mode="raw"):
            m32 = full_mask()
            m_lo = create_mask("h", 32)
            lane = varange(0, I32)

            zero_i = vdups(0, I32, mask=m32)
            one_i = vdups(1, I32, mask=m32)
            k8 = vdups(8, I32, mask=m32)
            k16 = vdups(16, I32, mask=m32)
            k_bias = vdups(E8M0_BIAS, I32, mask=m32)
            k_lane_wrap = vdups(VL - 1, I32, mask=m32)
            mant_mask = vdups(FP32_MANTISSA_MASK, I32, mask=m32)
            mag_mask = vdups(FP32_MAG_MASK, I32, mask=m32)

            # lane i -> (i + VL/2) % VL, the intra-register rotate-half permute
            rot_idx = vreinterpret(
                vbitwise_and(vadds(lane, VL // 2, mask=m32), k_lane_wrap, mask=m32), U32
            )
            # rotate_half(pe)[i] = -pe[i+Dr/2] for i < Dr/2 and +pe[i-Dr/2] above,
            # so the rotated register needs a sign flip on its low half.  This
            # used to be folded into sin_ub by negating it in place, but sin_ub is
            # a channel filled by DMA and read by VF; an extra in-place VF write
            # left its epoch ledger unresolvable.  One vmul against a loop
            # constant is cheaper than that pass was anyway.
            rope_sgn = vselect(
                vdups(1.0, F32, mask=m32),
                vdups(-1.0, F32, mask=m32),
                cond_mask=vges(lane, p.dr // 2, mask=m32),
            )
            # 1.0 on even lanes, 16.0 on odd lanes: pair-sum then packs two
            # nibbles.  f32 because asc_pair_reduce_sum only exists for f16/f32
            # -- there is no integer pair-reduce.  Nibbles are 0..15 and the
            # products 0..240, all exactly representable, so this is not a
            # precision compromise.
            odd = vne(vbitwise_and(lane, one_i, mask=m32), zero_i, mask=m32)
            pack_w = vselect(
                vdups(16.0, F32, mask=m32), vdups(1.0, F32, mask=m32), cond_mask=odd
            )
            # lanes 0..31 -> group g, lanes 32..63 -> group g+1
            gsel = vshr(lane, 5, mask=m32)

            # ---- pass 1: inplace RoPE, then |max| per 32-element MX group ----
            for r in range(rows):
                base = r * p.d
                for i in range_constexpr(p.n_reg):
                    x = vload(cv_ub, base + i * VL)
                    pe = i - (p.n_reg - p.rope_regs)
                    if const_expr(p.rope_regs > 0 and pe >= 0):
                        cosv = vload(self.cos_ub, r * p.dr + pe * VL)
                        sinv = vload(self.sin_ub, r * p.dr + pe * VL)
                        # Dr == 64 (enforced by Plan): rotate-half is a 32-lane
                        # rotation inside this single register, so it can be
                        # taken from x before x is overwritten.
                        rot = vmul(
                            vreinterpret(
                                vgather_reg(vreinterpret(x, I32), rot_idx), F32
                            ),
                            rope_sgn,
                            mask=m32,
                        )
                        # x*cos + rotate_half(x)*sin.  Operand order matters and
                        # is not what vmadd's docstring says: it lowers to
                        # asc_madd(dst, src0, src1) whose dst is in/out, so the
                        # hardware computes ``a*b + c`` for vmadd(a, b, c), not
                        # ``b*c + a``.  Passing the addend first silently
                        # computed x*cos*rot + sin, which on hardware showed up
                        # as an amax ~88^2 instead of ~88.
                        x = vmadd(
                            rot, sinv, vmul(x, cosv, mask=m32), mask=m32
                        )
                        vstore(cv_ub, base + i * VL, x, m32)
                    a = vabs(x, mask=m32)
                    # Both MX groups reduce under the same "lowest half" pset:
                    # the upper group is rotated down first (rot_idx is already
                    # live for the RoPE) so neither reduction depends on where a
                    # masked reduce parks its result for a high lane range.
                    a_hi = vreinterpret(
                        vgather_reg(vreinterpret(a, I32), rot_idx), F32
                    )
                    slot = r * p.groups + 2 * i
                    vstore_first(self.amax_ub, slot, vreduce_max(a, mask=m_lo))
                    vstore_first(self.amax_ub, slot + 1, vreduce_max(a_hi, mask=m_lo))

            # Pass 2 reads back the amax values pass 1 just wrote, and pass 3
            # reads back the RoPE result stored into cv_ub.  Inside one raw VF
            # region a UB store is not ordered against a later UB load, so
            # without this the last few rows of each subblock read stale data
            # (which is exactly how this showed up on hardware: only the rows at
            # the end of an AIV's range, and only for the first head).
            vmem_bar("vst_vld")

            # ---- pass 2: E8M0 scales for 64 groups at a time ----
            #
            # The golden takes ceil(log2(amax/6)) as
            # ``exp(fl(amax/6)) + (mant(fl(amax/6)) != 0)``.  Reproducing that
            # through a vector divide would need bit-exact IEEE division, which
            # the VPU reciprocal does not give, so use the algebraic identity
            # instead: with amax = 2^E * m, m in [1,2), amax/6 = 2^(E-3)*(m*4/3)
            # and m*4/3 in [4/3, 8/3), hence
            #     ceil(log2(amax/6)) = E - 2 + (m > 1.5).
            # Exact for every normal fp32 amax (checked against the golden over
            # the mantissa==0x400000 boundary and random bit patterns).
            for e in range(p.enc_regs):
                ab = vreinterpret(vload(self.amax_ub, e * VL), I32)
                exp_bits = vshr(ab, FP32_EXP_SHIFT, mask=m32)
                mant = vbitwise_and(ab, mant_mask, mask=m32)
                bump = vselect(
                    one_i, zero_i, cond_mask=vgts(mant, FP32_HALF_MANTISSA, mask=m32)
                )
                er = vadd(vadds(exp_bits, -2, mask=m32), bump, mask=m32)
                # amax < 1.5*2^-125 makes the golden's amax/6.0 land on an fp32
                # subnormal, where the identity above no longer holds; the golden
                # always yields 1 there, and er >= 1 is legal for every other
                # amax, so a clamp is enough.
                er = vmaxs(er, 1, mask=m32)
                # amax == 0 -> golden uses scale 1.0, i.e. the E8M0 bias.
                er = vselect(er, k_bias, cond_mask=vne(ab, zero_i, mask=m32))
                vstore(
                    self.off_ub,
                    e * VL,
                    vshl(vsub(er, k_bias, mask=m32), 3, mask=m32),
                    m32,
                )
                vstore_pack(
                    self.dsc_ub, e * VL, er, m32, pack_mode=PackMode.B32_TO_B8
                )

            # Pass 3 gathers the per-group offsets pass 2 just wrote.
            vmem_bar("vst_vld")

            # ---- pass 3: descale, round to E2M1, pack two nibbles per byte ----
            for r in range(rows):
                base = r * p.d
                gbase = vadds(gsel, r * p.groups, mask=m32)
                for i in range_constexpr(p.n_reg):
                    off = vgather(
                        self.off_ub,
                        vreinterpret(vadds(gbase, 2 * i, mask=m32), U32),
                        mask=m32,
                    )
                    xv = vload(cv_ub, base + i * VL)
                    mag = vbitwise_and(vreinterpret(xv, I32), mag_mask, mask=m32)
                    idx = vsub(vshr(mag, LUT_SHIFT, mask=m32), off, mask=m32)
                    # Only a low clamp: the high side is bounded analytically.
                    # er is derived from this group's own amax in pass 1 and
                    # ``2^(er-127) >= amax/6``, so |x|/scale <= 6 and idx <= 8*129
                    # + 4 = 1036 < LUT_LEN.  Clamping er up to 1 only enlarges
                    # the scale, which lowers idx.  Do not reuse this table with
                    # an externally supplied scale without adding vmins.
                    idx = vmaxs(idx, 0, mask=m32)
                    code = vgather(self.lut_ub, vreinterpret(idx, U32), mask=m32)
                    # Float compare, not the raw sign bit: the golden takes the
                    # sign as ``q < 0``, and -0.0 < 0 is false, so -0.0 must
                    # encode as nibble 0x0 rather than 0x8.  This matches the
                    # golden wherever the fp32 quotient is nonzero or x is +-0.
                    # It still differs when the quotient itself underflows to
                    # -0.0 while x != 0; making that exact needs the group's
                    # flush threshold max((er-150)<<23, 0) gathered alongside
                    # off, plus a compare and a select -- 3 more instructions on
                    # the path that is already the critical one, to cover a
                    # within-group dynamic range of 2^152 that an fp32
                    # accumulation cannot produce (see test_vf_math.py).
                    neg = vlts(xv, 0.0, mask=m32)
                    sign = vselect(k8, zero_i, cond_mask=neg)
                    nib = vbitwise_or(code, sign, mask=m32)
                    # even lane * 1 + odd lane * 16 -> one byte per lane pair,
                    # low nibble = even element, which is the golden's layout.
                    # Only VL/2 lanes carry a result, so the store is masked to
                    # the lowest half or it would clobber the next row.
                    nibf = vcast(nib, F32, mask=m32)
                    packed = vcast(
                        vpair_reduce_sum(vmul(nibf, pack_w, mask=m32), mask=m32),
                        I32,
                        mask=m32,
                    )
                    vstore_pack(
                        self.q_ub,
                        base // 2 + i * (VL // 2),
                        packed,
                        m_lo,
                        pack_mode=PackMode.B32_TO_B8,
                    )

    @jit
    def store(self, q_half, dsc_half, rows):
        """Write back exactly ``rows`` rows, so a short T tail cannot overrun GM."""
        p = self.p
        mem_copy(
            q_half,
            local_slice(self.q_ub, (rows, p.d // 2), stride=(p.d // 2, 1)),
        )
        mem_copy(
            dsc_half,
            local_slice(self.dsc_ub, (rows, p.groups), stride=(p.groups, 1)),
        )


# ---------------------------------------------------------------------------
# Fused kernel
# ---------------------------------------------------------------------------

@kernel
class attn_prolog_qwkv_kernel:
    def __init__(self, t, dim, q_lora, n_heads, d, dr):
        self.p = Plan(t, dim, q_lora, n_heads, d, dr)

    def __call__(
        self,
        x: Tensor,
        qr: Tensor,
        wqb: Tensor,
        ww: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        lut: Tensor,
        scale_gm: Tensor,
        q: Tensor,
        descale_q: Tensor,
        w: Tensor,
    ):
        p = self.p
        softmax_scale = scale_gm[0]
        cube_q = CubeQ(p)
        cube_w = CubeW(p)
        vec = VectorQ(p)
        # C_q -> V_q, the only cross-core edge. depth 2 lets the cube run head
        # h+1 while the vector still owns head h.
        cv_ub = Channel(
            MemLoc.UB,
            (p.rows_vec, p.d),
            F32,
            depth=D_CV,
            kind=ChannelKind.CrossCore,
        )
        subblock = get_subblock_id()

        n_work = p.n_tiles * p.n_head_blocks
        for work in range(get_block_idx(), n_work, get_block_num()):
            tile = work // p.n_head_blocks
            h_block = work % p.n_head_blocks
            # One tiler per distinct row width: the split is balanced over axis 0
            # only, so every tiler built from the same tile row count agrees on
            # where subblock 0 ends and subblock 1 begins.
            rows_full = tile_view(w, (p.base_m, p.n_heads), (tile, 0)).shape[0]
            split_cv = p.row_split(rows_full, p.d)
            split_q = p.row_split(rows_full, p.d // 2)
            split_dsc = p.row_split(rows_full, p.groups)
            rows = partition_view(
                tile_view(descale_q, (p.base_m, p.groups), (tile, 0)), split_dsc, subblock
            ).shape[0]

            if const_expr(p.dr > 0):
                split_rope = p.row_split(rows_full, p.dr)
                cos_half = partition_view(
                    tile_view(rope_cos, (p.base_m, p.dr), (tile, 0)), split_rope, subblock
                )
                sin_half = partition_view(
                    tile_view(rope_sin, (p.base_m, p.dr), (tile, 0)), split_rope, subblock
                )
            else:
                cos_half = None
                sin_half = None
            vec.load_tile_constants(lut, cos_half, sin_half)

            cube_q.load_qr(qr, descale_qr, tile)
            # ``w_leader_only`` is compile-time (depends on T).  The leader
            # path is the only one allowed to acquire the W L0C: a runtime
            # if around commit/release is read as a second writer.
            if const_expr(not p.w_leader_only) or h_block == 0:
                w_slot = cube_w.l0c.acquire()
                if const_expr(p.w_separate) and const_expr(PROBE != "no_w"):
                    cube_w.gemm_full(x, ww, tile, w_slot)
                    cube_w.store(w, tile, 0, softmax_scale, w_slot)
                    w_slot = None
                if const_expr(W_SCHED == "wpipe") and const_expr(PROBE != "no_w") and const_expr(not p.w_separate):
                    cube_w.fill(x, ww, tile, 0, 0)
                for local_head in range(p.heads_per_block):
                    head = h_block * p.heads_per_block + local_head
                    if (
                        const_expr(PROBE != "no_w")
                        and const_expr(W_SCHED == "wfirst")
                        and const_expr(not p.w_separate)
                    ):
                        cube_w.gemm_head_slice(
                            x, ww, tile, local_head, 0, w_slot
                        )
                    cube_q.gemm_head(wqb, descale_wqb, head)
                    if (
                        const_expr(PROBE != "no_w")
                        and const_expr(W_SCHED == "wafter")
                        and const_expr(not p.w_separate)
                    ):
                        cube_w.gemm_head_slice(
                            x, ww, tile, local_head, 0, w_slot
                        )
                    elif (
                        const_expr(PROBE != "no_w")
                        and const_expr(W_SCHED == "wpipe")
                        and const_expr(not p.w_separate)
                    ):
                        if local_head + 1 < p.heads_per_block:
                            cube_w.fill(x, ww, tile, local_head + 1, 0)
                        cube_w.drain(local_head, w_slot)
                    cube_q.drain_to_vec(cv_ub, split_cv)
                    vec.epilogue(cv_ub, rows)
                    vec.store(
                        partition_view(
                            tile_view(q, (p.base_m, p.d // 2), (tile, head)),
                            split_q,
                            subblock,
                        ),
                        partition_view(
                            tile_view(descale_q, (p.base_m, p.groups), (tile, head)),
                            split_dsc,
                            subblock,
                        ),
                        rows,
                    )
                if const_expr(PROBE != "no_w") and const_expr(not p.w_separate):
                    cube_w.store(w, tile, 0, softmax_scale, w_slot)
                elif const_expr(PROBE == "no_w"):
                    cube_w.l0c.commit(w_slot)
                    cube_w.l0c.release(cube_w.l0c.wait())
            else:
                for local_head in range(p.heads_per_block):
                    head = h_block * p.heads_per_block + local_head
                    cube_q.gemm_head(wqb, descale_wqb, head)
                    cube_q.drain_to_vec(cv_ub, split_cv)
                    vec.epilogue(cv_ub, rows)
                    vec.store(
                        partition_view(
                            tile_view(q, (p.base_m, p.d // 2), (tile, head)),
                            split_q,
                            subblock,
                        ),
                        partition_view(
                            tile_view(descale_q, (p.base_m, p.groups), (tile, head)),
                            split_dsc,
                            subblock,
                        ),
                        rows,
                    )


class AttnPrologQwkv:
    """Host wrapper: validates, plans tiles, launches the fused kernel."""

    def __init__(self, t, dim, q_lora, n_heads, d, dr):
        self.p = Plan(t, dim, q_lora, n_heads, d, dr)

    @jit
    def run(
        self,
        x: Tensor,
        qr: Tensor,
        wqb: Tensor,
        ww: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        lut: Tensor,
        scale_gm: Tensor,
        q: Tensor,
        descale_q: Tensor,
        w: Tensor,
    ):
        p = self.p
        attn_prolog_qwkv_kernel(p.t, p.dim, p.q_lora, p.n_heads, p.d, p.dr)[
            p.used_cores
        ](
            x,
            qr,
            wqb,
            ww,
            descale_qr,
            descale_wqb,
            rope_sin,
            rope_cos,
            lut,
            scale_gm,
            q,
            descale_q,
            w,
        )
