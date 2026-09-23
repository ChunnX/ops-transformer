# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CANNBotDSL implementation of the Engram residual gate."""

import math

import torch

from cannbotdsl import MemLoc, Tensor, const_expr, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.tensor import tile_slice
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg as rr
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy

__all__ = ["EngramGate", "EngramGateKernel", "engram_gate"]


VL = 64
DEFAULT_BLOCK_NUM = 56


@kernel
class EngramGateKernel:
    """One vector task per ``(token, hc)`` row."""

    def __init__(self, total_tokens: int, hc_mult: int, dim: int):
        from cannbotdsl.ops.info import get_mem_size

        self.total_tokens = int(total_tokens)
        self.hc_mult = int(hc_mult)
        self.dim = int(dim)
        self.num_full = self.dim // VL
        self.tail = self.dim - self.num_full * VL
        self.w = self.dim + (VL - self.tail) % VL

        ub_capacity = get_mem_size("ub")
        if self.w * 30 <= ub_capacity:
            self.chunked = False
            self.x_a = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.x_b = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.x_c = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.key_a = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.key_b = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.key_c = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.value_a = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.value_b = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.value_c = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.weight_ub = Buffer(MemLoc.UB, (self.w,), dtypes.float32)
            self.out_a = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.out_b = Buffer(MemLoc.UB, (self.w,), dtypes.bfloat16)
            self.x_fp32_ub = Buffer(MemLoc.UB, (self.w,), dtypes.float32)
            self.x_ub = self.x_a
            self.key_ub = self.key_a
            self.value_ub = self.value_a
            self.out_ub = self.out_a
        else:
            budget = ub_capacity - 8192
            max_chunk = budget // 12 // VL * VL
            if max_chunk < VL:
                raise RuntimeError(
                    f"EngramGateKernel needs at least one {VL}-element chunk "
                    f"(12 B/elem) but runtime reports {ub_capacity} B UB"
                )
            self.chunked = True
            self.chunk = min(self.w, int(max_chunk))
            self.num_chunks = (self.w + self.chunk - 1) // self.chunk
            self.last_len = self.w - (self.num_chunks - 1) * self.chunk
            self.last_gm_len = self.dim - (self.num_chunks - 1) * self.chunk
            self.last_full = self.last_gm_len // VL
            self.x_ch = Buffer(MemLoc.UB, (self.chunk,), dtypes.bfloat16)
            self.key_ch = Buffer(MemLoc.UB, (self.chunk,), dtypes.bfloat16)
            self.value_ch = Buffer(MemLoc.UB, (self.chunk,), dtypes.bfloat16)
            self.weight_ch = Buffer(MemLoc.UB, (self.chunk,), dtypes.float32)
            self.out_ch = Buffer(MemLoc.UB, (self.chunk,), dtypes.bfloat16)
            self.acc_x2 = Buffer(MemLoc.UB, (VL,), dtypes.float32)
            self.acc_k2 = Buffer(MemLoc.UB, (VL,), dtypes.float32)
            self.acc_dot = Buffer(MemLoc.UB, (VL,), dtypes.float32)
            self.gate_buf = Buffer(MemLoc.UB, (VL,), dtypes.float32)

    @jit
    def _compute(self, epsilon: dtypes.float32, clamp_value: dtypes.float32):
        dim = self.dim
        with vf(mode="raw"):
            full = rr.full_mask()
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            x_square_acc = rr.vdups(0.0, dtypes.float32, mask=full)
            key_square_acc = rr.vdups(0.0, dtypes.float32, mask=full)
            dot_acc = rr.vdups(0.0, dtypes.float32, mask=full)

            for segment in range(self.num_full):
                offset = segment * VL
                x = rr.vcast(
                    rr.vload_unpack(
                        self.x_ub, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                rr.vstore(self.x_fp32_ub, offset, x, full)
                key = rr.vcast(
                    rr.vload_unpack(
                        self.key_ub, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                weight = rr.vload(self.weight_ub, offset)
                x_square_acc = rr.vadd(
                    x_square_acc,
                    rr.vmul(x, x, mask=full),
                    mask=full,
                )
                key_square_acc = rr.vadd(
                    key_square_acc,
                    rr.vmul(key, key, mask=full),
                    mask=full,
                )
                dot_acc = rr.vadd(
                    dot_acc,
                    rr.vmul(rr.vmul(x, weight, mask=full), key, mask=full),
                    mask=full,
                )

            x_square_sum = rr.vreduce_sum(x_square_acc, mask=full)
            key_square_sum = rr.vreduce_sum(key_square_acc, mask=full)
            dot_sum = rr.vreduce_sum(dot_acc, mask=full)

            if const_expr(self.tail != 0):
                preg = rr.update_mask(self.tail, elem_bits=32)[0]
                tail_offset = self.num_full * VL
                x_t = rr.vcast(
                    rr.vload_unpack(
                        self.x_ub, tail_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=preg,
                )
                key_t = rr.vcast(
                    rr.vload_unpack(
                        self.key_ub, tail_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=preg,
                )
                weight_t = rr.vload(self.weight_ub, tail_offset)
                x_square_sum = rr.vadd(
                    x_square_sum,
                    rr.vreduce_sum(rr.vmul(x_t, x_t, mask=preg), mask=preg),
                    mask=lane0,
                )
                key_square_sum = rr.vadd(
                    key_square_sum,
                    rr.vreduce_sum(rr.vmul(key_t, key_t, mask=preg), mask=preg),
                    mask=lane0,
                )
                dot_sum = rr.vadd(
                    dot_sum,
                    rr.vreduce_sum(
                        rr.vmul(
                            rr.vmul(x_t, weight_t, mask=preg), key_t, mask=preg
                        ),
                        mask=preg,
                    ),
                    mask=lane0,
                )
                rr.vstore(self.x_fp32_ub, tail_offset, x_t, preg)

            one = rr.vdups(1.0, dtypes.float32, mask=lane0)
            x_rms = rr.vsqrt(
                rr.vadds(
                    rr.vmuls(x_square_sum, 1.0 / dim, mask=lane0),
                    epsilon,
                    mask=lane0,
                ),
                mask=lane0,
            )
            key_rms = rr.vsqrt(
                rr.vadds(
                    rr.vmuls(key_square_sum, 1.0 / dim, mask=lane0),
                    epsilon,
                    mask=lane0,
                ),
                mask=lane0,
            )
            normalized_dot = rr.vdiv(
                rr.vmuls(dot_sum, 1.0 / math.sqrt(dim), mask=lane0),
                rr.vmul(x_rms, key_rms, mask=lane0),
                mask=lane0,
            )

            magnitude = rr.vsqrt(
                rr.vmaxs(rr.vabs(normalized_dot, mask=lane0), clamp_value, mask=lane0),
                mask=lane0,
            )
            negative = rr.vlts(normalized_dot, 0.0, mask=lane0)
            signed_root = rr.vselect(
                rr.vmuls(magnitude, -1.0, mask=lane0),
                magnitude,
                cond_mask=negative,
            )
            gate = rr.vdiv(
                one,
                rr.vadd(
                    one,
                    rr.vexp(rr.vmuls(signed_root, -1.0, mask=lane0), mask=lane0),
                    mask=lane0,
                ),
                mask=lane0,
            )
            gate = rr.vdup(gate, mask=full)

            for segment in range(self.num_full):
                offset = segment * VL
                x = rr.vload(self.x_fp32_ub, offset)
                value = rr.vcast(
                    rr.vload_unpack(
                        self.value_ub, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                out = rr.vadd(x, rr.vmul(gate, value, mask=full), mask=full)
                rr.vstore_pack(
                    self.out_ub,
                    offset,
                    rr.vcast(out, dtypes.bfloat16, mask=full),
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            if const_expr(self.tail != 0):
                preg = rr.update_mask(self.tail, elem_bits=32)[0]
                tail_offset = self.num_full * VL
                x_t = rr.vload(self.x_fp32_ub, tail_offset)
                value_t = rr.vcast(
                    rr.vload_unpack(
                        self.value_ub, tail_offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=preg,
                )
                out_t = rr.vadd(x_t, rr.vmul(gate, value_t, mask=preg), mask=preg)
                rr.vstore_pack(
                    self.out_ub,
                    tail_offset,
                    rr.vcast(out_t, dtypes.bfloat16, mask=preg),
                    preg,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _acc_init(self):
        with vf(mode="raw"):
            full = rr.full_mask()
            zero = rr.vdups(0.0, dtypes.float32, mask=full)
            rr.vstore(self.acc_x2, 0, zero, full)
            rr.vstore(self.acc_k2, 0, zero, full)
            rr.vstore(self.acc_dot, 0, zero, full)

    @jit
    def _acc_full_chunk(self):
        with vf(mode="raw"):
            full = rr.full_mask()
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            x_acc = rr.vdups(0.0, dtypes.float32, mask=full)
            k_acc = rr.vdups(0.0, dtypes.float32, mask=full)
            d_acc = rr.vdups(0.0, dtypes.float32, mask=full)
            for segment in range(self.chunk // VL):
                offset = segment * VL
                x = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                key = rr.vcast(
                    rr.vload_unpack(
                        self.key_ch, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                weight = rr.vload(self.weight_ch, offset)
                x_acc = rr.vadd(x_acc, rr.vmul(x, x, mask=full), mask=full)
                k_acc = rr.vadd(k_acc, rr.vmul(key, key, mask=full), mask=full)
                d_acc = rr.vadd(
                    d_acc,
                    rr.vmul(rr.vmul(x, weight, mask=full), key, mask=full),
                    mask=full,
                )
            rr.vstore(
                self.acc_x2, 0,
                rr.vadd(rr.vload(self.acc_x2, 0), rr.vreduce_sum(x_acc, mask=full), mask=lane0),
                full,
            )
            rr.vstore(
                self.acc_k2, 0,
                rr.vadd(rr.vload(self.acc_k2, 0), rr.vreduce_sum(k_acc, mask=full), mask=lane0),
                full,
            )
            rr.vstore(
                self.acc_dot, 0,
                rr.vadd(rr.vload(self.acc_dot, 0), rr.vreduce_sum(d_acc, mask=full), mask=lane0),
                full,
            )

    @jit
    def _acc_last_chunk(self):
        with vf(mode="raw"):
            full = rr.full_mask()
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            x_acc = rr.vdups(0.0, dtypes.float32, mask=full)
            k_acc = rr.vdups(0.0, dtypes.float32, mask=full)
            d_acc = rr.vdups(0.0, dtypes.float32, mask=full)
            for segment in range(self.last_full):
                offset = segment * VL
                x = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                key = rr.vcast(
                    rr.vload_unpack(
                        self.key_ch, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                weight = rr.vload(self.weight_ch, offset)
                x_acc = rr.vadd(x_acc, rr.vmul(x, x, mask=full), mask=full)
                k_acc = rr.vadd(k_acc, rr.vmul(key, key, mask=full), mask=full)
                d_acc = rr.vadd(
                    d_acc,
                    rr.vmul(rr.vmul(x, weight, mask=full), key, mask=full),
                    mask=full,
                )
            if const_expr(self.tail != 0):
                preg = rr.update_mask(self.tail, elem_bits=32)[0]
                x_t = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, self.last_full * VL,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=preg,
                )
                key_t = rr.vcast(
                    rr.vload_unpack(
                        self.key_ch, self.last_full * VL,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=preg,
                )
                weight_t = rr.vload(self.weight_ch, self.last_full * VL)
                x_sum_t = rr.vreduce_sum(rr.vmul(x_t, x_t, mask=preg), mask=preg)
                k_sum_t = rr.vreduce_sum(rr.vmul(key_t, key_t, mask=preg), mask=preg)
                d_sum_t = rr.vreduce_sum(
                    rr.vmul(rr.vmul(x_t, weight_t, mask=preg), key_t, mask=preg),
                    mask=preg,
                )
                rr.vstore(
                    self.acc_x2, 0,
                    rr.vadd(
                        rr.vadd(rr.vload(self.acc_x2, 0), rr.vreduce_sum(x_acc, mask=full), mask=lane0),
                        x_sum_t, mask=lane0,
                    ),
                    full,
                )
                rr.vstore(
                    self.acc_k2, 0,
                    rr.vadd(
                        rr.vadd(rr.vload(self.acc_k2, 0), rr.vreduce_sum(k_acc, mask=full), mask=lane0),
                        k_sum_t, mask=lane0,
                    ),
                    full,
                )
                rr.vstore(
                    self.acc_dot, 0,
                    rr.vadd(
                        rr.vadd(rr.vload(self.acc_dot, 0), rr.vreduce_sum(d_acc, mask=full), mask=lane0),
                        d_sum_t, mask=lane0,
                    ),
                    full,
                )
            else:
                rr.vstore(
                    self.acc_x2, 0,
                    rr.vadd(rr.vload(self.acc_x2, 0), rr.vreduce_sum(x_acc, mask=full), mask=lane0),
                    full,
                )
                rr.vstore(
                    self.acc_k2, 0,
                    rr.vadd(rr.vload(self.acc_k2, 0), rr.vreduce_sum(k_acc, mask=full), mask=lane0),
                    full,
                )
                rr.vstore(
                    self.acc_dot, 0,
                    rr.vadd(rr.vload(self.acc_dot, 0), rr.vreduce_sum(d_acc, mask=full), mask=lane0),
                    full,
                )

    @jit
    def _finalize_gate(self, epsilon: dtypes.float32, clamp_value: dtypes.float32):
        dim = self.dim
        with vf(mode="raw"):
            full = rr.full_mask()
            lane0 = rr.update_mask(1, elem_bits=32)[0]
            x_square_sum = rr.vload(self.acc_x2, 0)
            key_square_sum = rr.vload(self.acc_k2, 0)
            dot_sum = rr.vload(self.acc_dot, 0)
            one = rr.vdups(1.0, dtypes.float32, mask=lane0)
            x_rms = rr.vsqrt(
                rr.vadds(rr.vmuls(x_square_sum, 1.0 / dim, mask=lane0), epsilon, mask=lane0),
                mask=lane0,
            )
            key_rms = rr.vsqrt(
                rr.vadds(rr.vmuls(key_square_sum, 1.0 / dim, mask=lane0), epsilon, mask=lane0),
                mask=lane0,
            )
            normalized_dot = rr.vdiv(
                rr.vmuls(dot_sum, 1.0 / math.sqrt(dim), mask=lane0),
                rr.vmul(x_rms, key_rms, mask=lane0),
                mask=lane0,
            )
            magnitude = rr.vsqrt(
                rr.vmaxs(rr.vabs(normalized_dot, mask=lane0), clamp_value, mask=lane0),
                mask=lane0,
            )
            negative = rr.vlts(normalized_dot, 0.0, mask=lane0)
            signed_root = rr.vselect(
                rr.vmuls(magnitude, -1.0, mask=lane0), magnitude, cond_mask=negative
            )
            gate = rr.vdiv(
                one,
                rr.vadd(
                    one,
                    rr.vexp(rr.vmuls(signed_root, -1.0, mask=lane0), mask=lane0),
                    mask=lane0,
                ),
                mask=lane0,
            )
            rr.vstore(self.gate_buf, 0, rr.vdup(gate, mask=full), full)

    @jit
    def _emit_full_chunk(self):
        with vf(mode="raw"):
            full = rr.full_mask()
            gate = rr.vload(self.gate_buf, 0)
            for segment in range(self.chunk // VL):
                offset = segment * VL
                x = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                value = rr.vcast(
                    rr.vload_unpack(
                        self.value_ch, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                out = rr.vadd(x, rr.vmul(gate, value, mask=full), mask=full)
                rr.vstore_pack(
                    self.out_ch,
                    offset,
                    rr.vcast(out, dtypes.bfloat16, mask=full),
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _emit_last_chunk(self):
        with vf(mode="raw"):
            full = rr.full_mask()
            gate = rr.vload(self.gate_buf, 0)
            for segment in range(self.last_full):
                offset = segment * VL
                x = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                value = rr.vcast(
                    rr.vload_unpack(
                        self.value_ch, offset, unpack_mode=rr.UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                out = rr.vadd(x, rr.vmul(gate, value, mask=full), mask=full)
                rr.vstore_pack(
                    self.out_ch,
                    offset,
                    rr.vcast(out, dtypes.bfloat16, mask=full),
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            if const_expr(self.tail != 0):
                preg = rr.update_mask(self.tail, elem_bits=32)[0]
                x_t = rr.vcast(
                    rr.vload_unpack(
                        self.x_ch, self.last_full * VL,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=preg,
                )
                value_t = rr.vcast(
                    rr.vload_unpack(
                        self.value_ch, self.last_full * VL,
                        unpack_mode=rr.UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=preg,
                )
                out_t = rr.vadd(x_t, rr.vmul(gate, value_t, mask=preg), mask=preg)
                rr.vstore_pack(
                    self.out_ch,
                    self.last_full * VL,
                    rr.vcast(out_t, dtypes.bfloat16, mask=preg),
                    preg,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )
            rr.vmem_bar("vst_vld")

    @jit
    def _init_rotation_buffers(self):
        with vf(mode="raw"):
            full = rr.full_mask()
            zero = rr.vcast(
                rr.vdups(0.0, dtypes.float32, mask=full),
                dtypes.bfloat16,
                mask=full,
            )
            rr.vstore_pack(self.x_b, 0, zero, full, pack_mode=rr.PackMode.B32_TO_B16)
            rr.vstore_pack(self.x_c, 0, zero, full, pack_mode=rr.PackMode.B32_TO_B16)
            rr.vstore_pack(self.key_b, 0, zero, full, pack_mode=rr.PackMode.B32_TO_B16)
            rr.vstore_pack(self.key_c, 0, zero, full, pack_mode=rr.PackMode.B32_TO_B16)
            rr.vstore_pack(self.value_b, 0, zero, full, pack_mode=rr.PackMode.B32_TO_B16)
            rr.vstore_pack(self.value_c, 0, zero, full, pack_mode=rr.PackMode.B32_TO_B16)

    def __call__(
        self,
        gm_x: Tensor,
        gm_key: Tensor,
        gm_value: Tensor,
        gm_weight: Tensor,
        gm_out: Tensor,
        epsilon: dtypes.float32,
        clamp_value: dtypes.float32,
        gm_mask: Tensor | None = None,
    ):
        block_idx = get_block_idx()
        block_num = get_block_num()
        rows = self.total_tokens * self.hc_mult
        if const_expr(self.chunked):
            for token_idx in range(block_idx, self.total_tokens, block_num):
                for hc_idx in dsl_range(
                    dtypes.int64(0), dtypes.int64(self.hc_mult), dtypes.int64(1)
                ):
                    if gm_mask is None:
                        x_row = gm_x[token_idx, hc_idx, None]
                        key_row = gm_key[token_idx, hc_idx, None]
                        w_row = gm_weight[hc_idx, None]
                        v_row = gm_value[token_idx, None]
                        o_row = gm_out[token_idx, hc_idx, None]
                        self._acc_init()
                        for ch in range(self.num_chunks):
                            mem_copy(self.x_ch, tile_slice(x_row, (1, self.chunk), (0, ch)))
                            mem_copy(self.key_ch, tile_slice(key_row, (1, self.chunk), (0, ch)))
                            mem_copy(self.weight_ch, tile_slice(w_row, (self.chunk,), (ch,)))
                            if ch + 1 < self.num_chunks:
                                self._acc_full_chunk()
                            else:
                                self._acc_last_chunk()
                        self._finalize_gate(epsilon, clamp_value)
                        for ch in range(self.num_chunks):
                            if self.num_chunks > 1:
                                mem_copy(self.x_ch, tile_slice(x_row, (1, self.chunk), (0, ch)))
                            mem_copy(self.value_ch, tile_slice(v_row, (self.chunk,), (ch,)))
                            if ch + 1 < self.num_chunks:
                                self._emit_full_chunk()
                            else:
                                self._emit_last_chunk()
                            mem_copy(tile_slice(o_row, (1, self.chunk), (0, ch)), self.out_ch)
                    else:
                        masked = gm_mask[token_idx]
                        if masked:
                            x_row = gm_x[token_idx, hc_idx, None]
                            o_row = gm_out[token_idx, hc_idx, None]
                            for ch in range(self.num_chunks):
                                mem_copy(self.x_ch, tile_slice(x_row, (1, self.chunk), (0, ch)))
                                mem_copy(tile_slice(o_row, (1, self.chunk), (0, ch)), self.x_ch)
                        else:
                            x_row = gm_x[token_idx, hc_idx, None]
                            key_row = gm_key[token_idx, hc_idx, None]
                            w_row = gm_weight[hc_idx, None]
                            v_row = gm_value[token_idx, None]
                            o_row = gm_out[token_idx, hc_idx, None]
                            self._acc_init()
                            for ch in range(self.num_chunks):
                                mem_copy(self.x_ch, tile_slice(x_row, (1, self.chunk), (0, ch)))
                                mem_copy(self.key_ch, tile_slice(key_row, (1, self.chunk), (0, ch)))
                                mem_copy(self.weight_ch, tile_slice(w_row, (self.chunk,), (ch,)))
                                if ch + 1 < self.num_chunks:
                                    self._acc_full_chunk()
                                else:
                                    self._acc_last_chunk()
                            self._finalize_gate(epsilon, clamp_value)
                            for ch in range(self.num_chunks):
                                if self.num_chunks > 1:
                                    mem_copy(self.x_ch, tile_slice(x_row, (1, self.chunk), (0, ch)))
                                mem_copy(self.value_ch, tile_slice(v_row, (self.chunk,), (ch,)))
                                if ch + 1 < self.num_chunks:
                                    self._emit_full_chunk()
                                else:
                                    self._emit_last_chunk()
                                mem_copy(tile_slice(o_row, (1, self.chunk), (0, ch)), self.out_ch)
            return
        self._init_rotation_buffers()
        if block_num % self.hc_mult == 0:
            hc_idx = block_idx % self.hc_mult
            first_row = block_idx
            pair_stride = 2 * block_num
            pairs = (rows - first_row + pair_stride - 1) // pair_stride
            if gm_mask is None:
                first_token = first_row // self.hc_mult
                mem_copy(self.x_a, gm_x[first_token, hc_idx, None])
                mem_copy(self.key_a, gm_key[first_token, hc_idx, None])
                mem_copy(self.weight_ub, gm_weight[hc_idx, None])
                mem_copy(self.value_a, gm_value[first_token, None])
                n_groups = pairs // 3
                for group_idx in range(n_groups):
                    base = first_row + group_idx * 3 * pair_stride
                    row0 = base
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    if row1 < rows:
                        mem_copy(self.value_b, gm_value[token1, None])
                        mem_copy(self.x_b, gm_x[token1, hc_idx, None])
                        mem_copy(self.key_b, gm_key[token1, hc_idx, None])
                    self.x_ub = self.x_a
                    self.key_ub = self.key_a
                    self.value_ub = self.value_a
                    self.out_ub = self.out_a
                    self._compute(epsilon, clamp_value)
                    mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    mem_copy(self.value_c, gm_value[token2, None])
                    mem_copy(self.x_c, gm_x[token2, hc_idx, None])
                    mem_copy(self.key_c, gm_key[token2, hc_idx, None])
                    if row1 < rows:
                        self.x_ub = self.x_b
                        self.key_ub = self.key_b
                        self.value_ub = self.value_b
                        self.out_ub = self.out_b
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                    row0 = base + pair_stride
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    if row1 < rows:
                        mem_copy(self.value_a, gm_value[token1, None])
                        mem_copy(self.x_a, gm_x[token1, hc_idx, None])
                        mem_copy(self.key_a, gm_key[token1, hc_idx, None])
                    self.x_ub = self.x_c
                    self.key_ub = self.key_c
                    self.value_ub = self.value_c
                    self.out_ub = self.out_a
                    self._compute(epsilon, clamp_value)
                    mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    mem_copy(self.value_b, gm_value[token2, None])
                    mem_copy(self.x_b, gm_x[token2, hc_idx, None])
                    mem_copy(self.key_b, gm_key[token2, hc_idx, None])
                    if row1 < rows:
                        self.x_ub = self.x_a
                        self.key_ub = self.key_a
                        self.value_ub = self.value_a
                        self.out_ub = self.out_b
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                    row0 = base + 2 * pair_stride
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    if row1 < rows:
                        mem_copy(self.value_c, gm_value[token1, None])
                        mem_copy(self.x_c, gm_x[token1, hc_idx, None])
                        mem_copy(self.key_c, gm_key[token1, hc_idx, None])
                    self.x_ub = self.x_b
                    self.key_ub = self.key_b
                    self.value_ub = self.value_b
                    self.out_ub = self.out_a
                    self._compute(epsilon, clamp_value)
                    mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    if row2 < rows:
                        mem_copy(self.value_a, gm_value[token2, None])
                        mem_copy(self.x_a, gm_x[token2, hc_idx, None])
                        mem_copy(self.key_a, gm_key[token2, hc_idx, None])
                    if row1 < rows:
                        self.x_ub = self.x_c
                        self.key_ub = self.key_c
                        self.value_ub = self.value_c
                        self.out_ub = self.out_b
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                rem = pairs - 3 * n_groups
                if rem >= 1:
                    base = first_row + n_groups * 3 * pair_stride
                    row0 = base
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    if row1 < rows:
                        mem_copy(self.value_b, gm_value[token1, None])
                        mem_copy(self.x_b, gm_x[token1, hc_idx, None])
                        mem_copy(self.key_b, gm_key[token1, hc_idx, None])
                    self.x_ub = self.x_a
                    self.key_ub = self.key_a
                    self.value_ub = self.value_a
                    self.out_ub = self.out_a
                    self._compute(epsilon, clamp_value)
                    mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    if row1 < rows:
                        self.x_ub = self.x_b
                        self.key_ub = self.key_b
                        self.value_ub = self.value_b
                        self.out_ub = self.out_b
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                    if rem >= 2:
                        row0 = base + pair_stride
                        row1 = row0 + block_num
                        token0 = row0 // self.hc_mult
                        token1 = row1 // self.hc_mult
                        mem_copy(self.value_c, gm_value[token2, None])
                        mem_copy(self.x_c, gm_x[token2, hc_idx, None])
                        mem_copy(self.key_c, gm_key[token2, hc_idx, None])
                        if row1 < rows:
                            mem_copy(self.value_a, gm_value[token1, None])
                            mem_copy(self.x_a, gm_x[token1, hc_idx, None])
                            mem_copy(self.key_a, gm_key[token1, hc_idx, None])
                        self.x_ub = self.x_c
                        self.key_ub = self.key_c
                        self.value_ub = self.value_c
                        self.out_ub = self.out_a
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                        if row1 < rows:
                            self.x_ub = self.x_a
                            self.key_ub = self.key_a
                            self.value_ub = self.value_a
                            self.out_ub = self.out_b
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
            else:
                first_token = first_row // self.hc_mult
                first_masked = gm_mask[first_token]
                if first_masked:
                    mem_copy(self.x_a, gm_x[first_token, hc_idx, None])
                    mem_copy(self.weight_ub, gm_weight[hc_idx, None])
                else:
                    mem_copy(self.x_a, gm_x[first_token, hc_idx, None])
                    mem_copy(self.key_a, gm_key[first_token, hc_idx, None])
                    mem_copy(self.weight_ub, gm_weight[hc_idx, None])
                    mem_copy(self.value_a, gm_value[first_token, None])
                n_groups = pairs // 3
                for group_idx in range(n_groups):
                    base = first_row + group_idx * 3 * pair_stride
                    row0 = base
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    if row1 < rows:
                        masked1 = gm_mask[token1]
                        if masked1:
                            mem_copy(self.x_b, gm_x[token1, hc_idx, None])
                        else:
                            mem_copy(self.value_b, gm_value[token1, None])
                            mem_copy(self.x_b, gm_x[token1, hc_idx, None])
                            mem_copy(self.key_b, gm_key[token1, hc_idx, None])
                    self.x_ub = self.x_a
                    self.key_ub = self.key_a
                    self.value_ub = self.value_a
                    self.out_ub = self.out_a
                    masked0 = gm_mask[token0]
                    if masked0:
                        mem_copy(self.out_ub, self.x_ub)
                        mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    else:
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    masked2 = gm_mask[token2]
                    if masked2:
                        mem_copy(self.x_c, gm_x[token2, hc_idx, None])
                    else:
                        mem_copy(self.value_c, gm_value[token2, None])
                        mem_copy(self.x_c, gm_x[token2, hc_idx, None])
                        mem_copy(self.key_c, gm_key[token2, hc_idx, None])
                    if row1 < rows:
                        masked1b = gm_mask[token1]
                        self.x_ub = self.x_b
                        self.key_ub = self.key_b
                        self.value_ub = self.value_b
                        self.out_ub = self.out_b
                        if masked1b:
                            mem_copy(self.out_ub, self.x_ub)
                            mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                        else:
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                    row0 = base + pair_stride
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    if row1 < rows:
                        masked1 = gm_mask[token1]
                        if masked1:
                            mem_copy(self.x_a, gm_x[token1, hc_idx, None])
                        else:
                            mem_copy(self.value_a, gm_value[token1, None])
                            mem_copy(self.x_a, gm_x[token1, hc_idx, None])
                            mem_copy(self.key_a, gm_key[token1, hc_idx, None])
                    self.x_ub = self.x_c
                    self.key_ub = self.key_c
                    self.value_ub = self.value_c
                    self.out_ub = self.out_a
                    masked0 = gm_mask[token0]
                    if masked0:
                        mem_copy(self.out_ub, self.x_ub)
                        mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    else:
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    masked2 = gm_mask[token2]
                    if masked2:
                        mem_copy(self.x_b, gm_x[token2, hc_idx, None])
                    else:
                        mem_copy(self.value_b, gm_value[token2, None])
                        mem_copy(self.x_b, gm_x[token2, hc_idx, None])
                        mem_copy(self.key_b, gm_key[token2, hc_idx, None])
                    if row1 < rows:
                        masked1b = gm_mask[token1]
                        self.x_ub = self.x_a
                        self.key_ub = self.key_a
                        self.value_ub = self.value_a
                        self.out_ub = self.out_b
                        if masked1b:
                            mem_copy(self.out_ub, self.x_ub)
                            mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                        else:
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                    row0 = base + 2 * pair_stride
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    if row1 < rows:
                        masked1 = gm_mask[token1]
                        if masked1:
                            mem_copy(self.x_c, gm_x[token1, hc_idx, None])
                        else:
                            mem_copy(self.value_c, gm_value[token1, None])
                            mem_copy(self.x_c, gm_x[token1, hc_idx, None])
                            mem_copy(self.key_c, gm_key[token1, hc_idx, None])
                    self.x_ub = self.x_b
                    self.key_ub = self.key_b
                    self.value_ub = self.value_b
                    self.out_ub = self.out_a
                    masked0 = gm_mask[token0]
                    if masked0:
                        mem_copy(self.out_ub, self.x_ub)
                        mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    else:
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    if row2 < rows:
                        masked2 = gm_mask[token2]
                        if masked2:
                            mem_copy(self.x_a, gm_x[token2, hc_idx, None])
                        else:
                            mem_copy(self.value_a, gm_value[token2, None])
                            mem_copy(self.x_a, gm_x[token2, hc_idx, None])
                            mem_copy(self.key_a, gm_key[token2, hc_idx, None])
                    if row1 < rows:
                        masked1b = gm_mask[token1]
                        self.x_ub = self.x_c
                        self.key_ub = self.key_c
                        self.value_ub = self.value_c
                        self.out_ub = self.out_b
                        if masked1b:
                            mem_copy(self.out_ub, self.x_ub)
                            mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                        else:
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                rem = pairs - 3 * n_groups
                if rem >= 1:
                    base = first_row + n_groups * 3 * pair_stride
                    row0 = base
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    if row1 < rows:
                        masked1 = gm_mask[token1]
                        if masked1:
                            mem_copy(self.x_b, gm_x[token1, hc_idx, None])
                        else:
                            mem_copy(self.value_b, gm_value[token1, None])
                            mem_copy(self.x_b, gm_x[token1, hc_idx, None])
                            mem_copy(self.key_b, gm_key[token1, hc_idx, None])
                    self.x_ub = self.x_a
                    self.key_ub = self.key_a
                    self.value_ub = self.value_a
                    self.out_ub = self.out_a
                    masked0 = gm_mask[token0]
                    if masked0:
                        mem_copy(self.out_ub, self.x_ub)
                        mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    else:
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                    if row1 < rows:
                        masked1b = gm_mask[token1]
                        self.x_ub = self.x_b
                        self.key_ub = self.key_b
                        self.value_ub = self.value_b
                        self.out_ub = self.out_b
                        if masked1b:
                            mem_copy(self.out_ub, self.x_ub)
                            mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                        else:
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                    if rem >= 2:
                        row0 = base + pair_stride
                        row1 = row0 + block_num
                        token0 = row0 // self.hc_mult
                        token1 = row1 // self.hc_mult
                        masked2 = gm_mask[token2]
                        if masked2:
                            mem_copy(self.x_c, gm_x[token2, hc_idx, None])
                        else:
                            mem_copy(self.value_c, gm_value[token2, None])
                            mem_copy(self.x_c, gm_x[token2, hc_idx, None])
                            mem_copy(self.key_c, gm_key[token2, hc_idx, None])
                        if row1 < rows:
                            masked1 = gm_mask[token1]
                            if masked1:
                                mem_copy(self.x_a, gm_x[token1, hc_idx, None])
                            else:
                                mem_copy(self.value_a, gm_value[token1, None])
                                mem_copy(self.x_a, gm_x[token1, hc_idx, None])
                                mem_copy(self.key_a, gm_key[token1, hc_idx, None])
                        self.x_ub = self.x_c
                        self.key_ub = self.key_c
                        self.value_ub = self.value_c
                        self.out_ub = self.out_a
                        masked0 = gm_mask[token0]
                        if masked0:
                            mem_copy(self.out_ub, self.x_ub)
                            mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                        else:
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token0, hc_idx, None], self.out_ub)
                        if row1 < rows:
                            masked1b = gm_mask[token1]
                            self.x_ub = self.x_a
                            self.key_ub = self.key_a
                            self.value_ub = self.value_a
                            self.out_ub = self.out_b
                            if masked1b:
                                mem_copy(self.out_ub, self.x_ub)
                                mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
                            else:
                                self._compute(epsilon, clamp_value)
                                mem_copy(gm_out[token1, hc_idx, None], self.out_ub)
        else:
            first_row = block_idx
            pair_stride = 2 * block_num
            pairs = (rows - first_row + pair_stride - 1) // pair_stride
            first_token = first_row // self.hc_mult
            first_hc = first_row - first_token * self.hc_mult
            if gm_mask is None:
                mem_copy(self.x_a, gm_x[first_token, first_hc, None])
                mem_copy(self.key_a, gm_key[first_token, first_hc, None])
                mem_copy(self.weight_ub, gm_weight[first_hc, None])
                mem_copy(self.value_a, gm_value[first_token, None])
                n_groups = pairs // 3
                for group_idx in range(n_groups):
                    base = first_row + group_idx * 3 * pair_stride
                    row0 = base
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    hc0 = row0 - token0 * self.hc_mult
                    hc1 = row1 - token1 * self.hc_mult
                    hc2 = row2 - token2 * self.hc_mult
                    if row1 < rows:
                        mem_copy(self.value_b, gm_value[token1, None])
                        mem_copy(self.x_b, gm_x[token1, hc1, None])
                        mem_copy(self.key_b, gm_key[token1, hc1, None])
                    mem_copy(self.weight_ub, gm_weight[hc0, None])
                    self.x_ub = self.x_a
                    self.key_ub = self.key_a
                    self.value_ub = self.value_a
                    self.out_ub = self.out_a
                    self._compute(epsilon, clamp_value)
                    mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    mem_copy(self.value_c, gm_value[token2, None])
                    mem_copy(self.x_c, gm_x[token2, hc2, None])
                    mem_copy(self.key_c, gm_key[token2, hc2, None])
                    if row1 < rows:
                        mem_copy(self.weight_ub, gm_weight[hc1, None])
                        self.x_ub = self.x_b
                        self.key_ub = self.key_b
                        self.value_ub = self.value_b
                        self.out_ub = self.out_b
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token1, hc1, None], self.out_ub)
                    row0 = base + pair_stride
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    hc0 = row0 - token0 * self.hc_mult
                    hc1 = row1 - token1 * self.hc_mult
                    hc2 = row2 - token2 * self.hc_mult
                    if row1 < rows:
                        mem_copy(self.value_a, gm_value[token1, None])
                        mem_copy(self.x_a, gm_x[token1, hc1, None])
                        mem_copy(self.key_a, gm_key[token1, hc1, None])
                    mem_copy(self.weight_ub, gm_weight[hc0, None])
                    self.x_ub = self.x_c
                    self.key_ub = self.key_c
                    self.value_ub = self.value_c
                    self.out_ub = self.out_a
                    self._compute(epsilon, clamp_value)
                    mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    mem_copy(self.value_b, gm_value[token2, None])
                    mem_copy(self.x_b, gm_x[token2, hc2, None])
                    mem_copy(self.key_b, gm_key[token2, hc2, None])
                    if row1 < rows:
                        mem_copy(self.weight_ub, gm_weight[hc1, None])
                        self.x_ub = self.x_a
                        self.key_ub = self.key_a
                        self.value_ub = self.value_a
                        self.out_ub = self.out_b
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token1, hc1, None], self.out_ub)
                    row0 = base + 2 * pair_stride
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    hc0 = row0 - token0 * self.hc_mult
                    hc1 = row1 - token1 * self.hc_mult
                    hc2 = row2 - token2 * self.hc_mult
                    if row1 < rows:
                        mem_copy(self.value_c, gm_value[token1, None])
                        mem_copy(self.x_c, gm_x[token1, hc1, None])
                        mem_copy(self.key_c, gm_key[token1, hc1, None])
                    mem_copy(self.weight_ub, gm_weight[hc0, None])
                    self.x_ub = self.x_b
                    self.key_ub = self.key_b
                    self.value_ub = self.value_b
                    self.out_ub = self.out_a
                    self._compute(epsilon, clamp_value)
                    mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    if row2 < rows:
                        mem_copy(self.value_a, gm_value[token2, None])
                        mem_copy(self.x_a, gm_x[token2, hc2, None])
                        mem_copy(self.key_a, gm_key[token2, hc2, None])
                    if row1 < rows:
                        mem_copy(self.weight_ub, gm_weight[hc1, None])
                        self.x_ub = self.x_c
                        self.key_ub = self.key_c
                        self.value_ub = self.value_c
                        self.out_ub = self.out_b
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token1, hc1, None], self.out_ub)
                rem = pairs - 3 * n_groups
                if rem >= 1:
                    base = first_row + n_groups * 3 * pair_stride
                    row0 = base
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    hc0 = row0 - token0 * self.hc_mult
                    hc1 = row1 - token1 * self.hc_mult
                    hc2 = row2 - token2 * self.hc_mult
                    if row1 < rows:
                        mem_copy(self.value_b, gm_value[token1, None])
                        mem_copy(self.x_b, gm_x[token1, hc1, None])
                        mem_copy(self.key_b, gm_key[token1, hc1, None])
                    mem_copy(self.weight_ub, gm_weight[hc0, None])
                    self.x_ub = self.x_a
                    self.key_ub = self.key_a
                    self.value_ub = self.value_a
                    self.out_ub = self.out_a
                    self._compute(epsilon, clamp_value)
                    mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    if row1 < rows:
                        mem_copy(self.weight_ub, gm_weight[hc1, None])
                        self.x_ub = self.x_b
                        self.key_ub = self.key_b
                        self.value_ub = self.value_b
                        self.out_ub = self.out_b
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token1, hc1, None], self.out_ub)
                    if rem >= 2:
                        row0 = base + pair_stride
                        row1 = row0 + block_num
                        token0 = row0 // self.hc_mult
                        token1 = row1 // self.hc_mult
                        hc0 = row0 - token0 * self.hc_mult
                        hc1 = row1 - token1 * self.hc_mult
                        mem_copy(self.value_c, gm_value[token2, None])
                        mem_copy(self.x_c, gm_x[token2, hc2, None])
                        mem_copy(self.key_c, gm_key[token2, hc2, None])
                        if row1 < rows:
                            mem_copy(self.value_a, gm_value[token1, None])
                            mem_copy(self.x_a, gm_x[token1, hc1, None])
                            mem_copy(self.key_a, gm_key[token1, hc1, None])
                        mem_copy(self.weight_ub, gm_weight[hc0, None])
                        self.x_ub = self.x_c
                        self.key_ub = self.key_c
                        self.value_ub = self.value_c
                        self.out_ub = self.out_a
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token0, hc0, None], self.out_ub)
                        if row1 < rows:
                            mem_copy(self.weight_ub, gm_weight[hc1, None])
                            self.x_ub = self.x_a
                            self.key_ub = self.key_a
                            self.value_ub = self.value_a
                            self.out_ub = self.out_b
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token1, hc1, None], self.out_ub)
            else:
                first_masked = gm_mask[first_token]
                if first_masked:
                    mem_copy(self.x_a, gm_x[first_token, first_hc, None])
                else:
                    mem_copy(self.x_a, gm_x[first_token, first_hc, None])
                    mem_copy(self.key_a, gm_key[first_token, first_hc, None])
                    mem_copy(self.weight_ub, gm_weight[first_hc, None])
                    mem_copy(self.value_a, gm_value[first_token, None])
                n_groups = pairs // 3
                for group_idx in range(n_groups):
                    base = first_row + group_idx * 3 * pair_stride
                    row0 = base
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    hc0 = row0 - token0 * self.hc_mult
                    hc1 = row1 - token1 * self.hc_mult
                    hc2 = row2 - token2 * self.hc_mult
                    if row1 < rows:
                        masked1 = gm_mask[token1]
                        if masked1:
                            mem_copy(self.x_b, gm_x[token1, hc1, None])
                        else:
                            mem_copy(self.value_b, gm_value[token1, None])
                            mem_copy(self.x_b, gm_x[token1, hc1, None])
                            mem_copy(self.key_b, gm_key[token1, hc1, None])
                    self.x_ub = self.x_a
                    self.key_ub = self.key_a
                    self.value_ub = self.value_a
                    self.out_ub = self.out_a
                    masked0 = gm_mask[token0]
                    if masked0:
                        mem_copy(self.out_ub, self.x_ub)
                        mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    else:
                        mem_copy(self.weight_ub, gm_weight[hc0, None])
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    masked2 = gm_mask[token2]
                    if masked2:
                        mem_copy(self.x_c, gm_x[token2, hc2, None])
                    else:
                        mem_copy(self.value_c, gm_value[token2, None])
                        mem_copy(self.x_c, gm_x[token2, hc2, None])
                        mem_copy(self.key_c, gm_key[token2, hc2, None])
                    if row1 < rows:
                        masked1b = gm_mask[token1]
                        self.x_ub = self.x_b
                        self.key_ub = self.key_b
                        self.value_ub = self.value_b
                        self.out_ub = self.out_b
                        if masked1b:
                            mem_copy(self.out_ub, self.x_ub)
                            mem_copy(gm_out[token1, hc1, None], self.out_ub)
                        else:
                            mem_copy(self.weight_ub, gm_weight[hc1, None])
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token1, hc1, None], self.out_ub)
                    row0 = base + pair_stride
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    hc0 = row0 - token0 * self.hc_mult
                    hc1 = row1 - token1 * self.hc_mult
                    hc2 = row2 - token2 * self.hc_mult
                    if row1 < rows:
                        masked1 = gm_mask[token1]
                        if masked1:
                            mem_copy(self.x_a, gm_x[token1, hc1, None])
                        else:
                            mem_copy(self.value_a, gm_value[token1, None])
                            mem_copy(self.x_a, gm_x[token1, hc1, None])
                            mem_copy(self.key_a, gm_key[token1, hc1, None])
                    self.x_ub = self.x_c
                    self.key_ub = self.key_c
                    self.value_ub = self.value_c
                    self.out_ub = self.out_a
                    masked0 = gm_mask[token0]
                    if masked0:
                        mem_copy(self.out_ub, self.x_ub)
                        mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    else:
                        mem_copy(self.weight_ub, gm_weight[hc0, None])
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    masked2 = gm_mask[token2]
                    if masked2:
                        mem_copy(self.x_b, gm_x[token2, hc2, None])
                    else:
                        mem_copy(self.value_b, gm_value[token2, None])
                        mem_copy(self.x_b, gm_x[token2, hc2, None])
                        mem_copy(self.key_b, gm_key[token2, hc2, None])
                    if row1 < rows:
                        masked1b = gm_mask[token1]
                        self.x_ub = self.x_a
                        self.key_ub = self.key_a
                        self.value_ub = self.value_a
                        self.out_ub = self.out_b
                        if masked1b:
                            mem_copy(self.out_ub, self.x_ub)
                            mem_copy(gm_out[token1, hc1, None], self.out_ub)
                        else:
                            mem_copy(self.weight_ub, gm_weight[hc1, None])
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token1, hc1, None], self.out_ub)
                    row0 = base + 2 * pair_stride
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    hc0 = row0 - token0 * self.hc_mult
                    hc1 = row1 - token1 * self.hc_mult
                    hc2 = row2 - token2 * self.hc_mult
                    if row1 < rows:
                        masked1 = gm_mask[token1]
                        if masked1:
                            mem_copy(self.x_c, gm_x[token1, hc1, None])
                        else:
                            mem_copy(self.value_c, gm_value[token1, None])
                            mem_copy(self.x_c, gm_x[token1, hc1, None])
                            mem_copy(self.key_c, gm_key[token1, hc1, None])
                    self.x_ub = self.x_b
                    self.key_ub = self.key_b
                    self.value_ub = self.value_b
                    self.out_ub = self.out_a
                    masked0 = gm_mask[token0]
                    if masked0:
                        mem_copy(self.out_ub, self.x_ub)
                        mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    else:
                        mem_copy(self.weight_ub, gm_weight[hc0, None])
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    if row2 < rows:
                        masked2 = gm_mask[token2]
                        if masked2:
                            mem_copy(self.x_a, gm_x[token2, hc2, None])
                        else:
                            mem_copy(self.value_a, gm_value[token2, None])
                            mem_copy(self.x_a, gm_x[token2, hc2, None])
                            mem_copy(self.key_a, gm_key[token2, hc2, None])
                    if row1 < rows:
                        masked1b = gm_mask[token1]
                        self.x_ub = self.x_c
                        self.key_ub = self.key_c
                        self.value_ub = self.value_c
                        self.out_ub = self.out_b
                        if masked1b:
                            mem_copy(self.out_ub, self.x_ub)
                            mem_copy(gm_out[token1, hc1, None], self.out_ub)
                        else:
                            mem_copy(self.weight_ub, gm_weight[hc1, None])
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token1, hc1, None], self.out_ub)
                rem = pairs - 3 * n_groups
                if rem >= 1:
                    base = first_row + n_groups * 3 * pair_stride
                    row0 = base
                    row1 = row0 + block_num
                    row2 = row0 + pair_stride
                    token0 = row0 // self.hc_mult
                    token1 = row1 // self.hc_mult
                    token2 = row2 // self.hc_mult
                    hc0 = row0 - token0 * self.hc_mult
                    hc1 = row1 - token1 * self.hc_mult
                    hc2 = row2 - token2 * self.hc_mult
                    if row1 < rows:
                        masked1 = gm_mask[token1]
                        if masked1:
                            mem_copy(self.x_b, gm_x[token1, hc1, None])
                        else:
                            mem_copy(self.value_b, gm_value[token1, None])
                            mem_copy(self.x_b, gm_x[token1, hc1, None])
                            mem_copy(self.key_b, gm_key[token1, hc1, None])
                    self.x_ub = self.x_a
                    self.key_ub = self.key_a
                    self.value_ub = self.value_a
                    self.out_ub = self.out_a
                    masked0 = gm_mask[token0]
                    if masked0:
                        mem_copy(self.out_ub, self.x_ub)
                        mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    else:
                        mem_copy(self.weight_ub, gm_weight[hc0, None])
                        self._compute(epsilon, clamp_value)
                        mem_copy(gm_out[token0, hc0, None], self.out_ub)
                    if row1 < rows:
                        masked1b = gm_mask[token1]
                        self.x_ub = self.x_b
                        self.key_ub = self.key_b
                        self.value_ub = self.value_b
                        self.out_ub = self.out_b
                        if masked1b:
                            mem_copy(self.out_ub, self.x_ub)
                            mem_copy(gm_out[token1, hc1, None], self.out_ub)
                        else:
                            mem_copy(self.weight_ub, gm_weight[hc1, None])
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token1, hc1, None], self.out_ub)
                    if rem >= 2:
                        row0 = base + pair_stride
                        row1 = row0 + block_num
                        token0 = row0 // self.hc_mult
                        token1 = row1 // self.hc_mult
                        hc0 = row0 - token0 * self.hc_mult
                        hc1 = row1 - token1 * self.hc_mult
                        masked2 = gm_mask[token2]
                        if masked2:
                            mem_copy(self.x_c, gm_x[token2, hc2, None])
                        else:
                            mem_copy(self.value_c, gm_value[token2, None])
                            mem_copy(self.x_c, gm_x[token2, hc2, None])
                            mem_copy(self.key_c, gm_key[token2, hc2, None])
                        if row1 < rows:
                            masked1 = gm_mask[token1]
                            if masked1:
                                mem_copy(self.x_a, gm_x[token1, hc1, None])
                            else:
                                mem_copy(self.value_a, gm_value[token1, None])
                                mem_copy(self.x_a, gm_x[token1, hc1, None])
                                mem_copy(self.key_a, gm_key[token1, hc1, None])
                        self.x_ub = self.x_c
                        self.key_ub = self.key_c
                        self.value_ub = self.value_c
                        self.out_ub = self.out_a
                        masked0 = gm_mask[token0]
                        if masked0:
                            mem_copy(self.out_ub, self.x_ub)
                            mem_copy(gm_out[token0, hc0, None], self.out_ub)
                        else:
                            mem_copy(self.weight_ub, gm_weight[hc0, None])
                            self._compute(epsilon, clamp_value)
                            mem_copy(gm_out[token0, hc0, None], self.out_ub)
                        if row1 < rows:
                            masked1b = gm_mask[token1]
                            self.x_ub = self.x_a
                            self.key_ub = self.key_a
                            self.value_ub = self.value_a
                            self.out_ub = self.out_b
                            if masked1b:
                                mem_copy(self.out_ub, self.x_ub)
                                mem_copy(gm_out[token1, hc1, None], self.out_ub)
                            else:
                                mem_copy(self.weight_ub, gm_weight[hc1, None])
                                self._compute(epsilon, clamp_value)
                                mem_copy(gm_out[token1, hc1, None], self.out_ub)
class EngramGate:
    """Host-side CANNBotDSL launcher for one static Engram shape."""

    def __init__(
        self,
        eps: float,
        clamp_value: float,
        dim: int = 5120,
        hc_mult: int = 4,
    ):
        self.dim = int(dim)
        self.hc_mult = int(hc_mult)
        self.eps = float(eps)
        self.clamp_value = float(clamp_value)

    @jit
    def run(
        self,
        gm_x,
        gm_key,
        gm_value,
        gm_weight,
        gm_out,
        mask: Tensor | None = None,
    ):
        total_tokens = gm_x.shape[0]
        if const_expr(
            len(gm_x.shape) != 3
            or gm_x.shape[1] != self.hc_mult
            or gm_x.shape[2] != self.dim
        ):
            raise ValueError(
                f"x must have shape [T, {self.hc_mult}, {self.dim}] to match "
                "this EngramGate instance"
            )
        if const_expr(
            tuple(gm_key.shape) != tuple(gm_x.shape)
            or tuple(gm_out.shape) != tuple(gm_x.shape)
            or tuple(gm_value.shape) != (total_tokens, self.dim)
            or tuple(gm_weight.shape) != (self.hc_mult, self.dim)
        ):
            raise ValueError(
                "key and out must match x; value must have shape "
                f"[{total_tokens}, {self.dim}]; weight must have shape "
                f"[{self.hc_mult}, {self.dim}]"
            )
        if const_expr(mask is not None and mask.shape[0] != total_tokens):
            raise ValueError(f"mask must have shape [{total_tokens}] to match x")
        rows = total_tokens * self.hc_mult
        block_num = min(DEFAULT_BLOCK_NUM, rows)
        op = EngramGateKernel(total_tokens, self.hc_mult, self.dim)
        if mask is not None:
            op[block_num](
                gm_x,
                gm_key,
                gm_value,
                gm_weight,
                gm_out,
                self.eps,
                self.clamp_value,
                mask,
            )
        else:
            op[block_num](
                gm_x,
                gm_key,
                gm_value,
                gm_weight,
                gm_out,
                self.eps,
                self.clamp_value,
            )

    def __call__(self, x, key, value, weight, image_mask=None):
        return engram_gate(
            x,
            key,
            value,
            weight,
            image_mask=image_mask,
            eps=self.eps,
            clamp_value=self.clamp_value,
        )


def engram_gate(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    image_mask: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-6,
    clamp_value: float = 1.0e-6,
) -> torch.Tensor:
    """Apply the Engram gate and return a BF16 residual tensor.

    ``x``/``key`` accept any ndim >= 2: the last dim is ``dim``, the
    second-to-last is ``hc_mult`` (a 2-D input means ``hc_mult == 1``), and
    all leading dims flatten into tokens. ``value`` and ``image_mask`` follow
    the leading dims; the output keeps ``x``'s shape.
    """
    if x.ndim < 2:
        raise ValueError("x must have shape [..., hc_mult, dim] with ndim >= 2")
    if x.ndim == 2:
        lead = x.shape[:-1]
        hc_mult = 1
    else:
        lead = x.shape[:-2]
        hc_mult = int(x.shape[-2])
    dim = int(x.shape[-1])
    total_tokens = 1
    for size in lead:
        total_tokens *= int(size)
    if total_tokens <= 0 or hc_mult <= 0 or dim <= 0:
        raise ValueError("x dimensions must be positive")
    if tuple(key.shape) != tuple(x.shape):
        raise ValueError("key must have the same shape as x")
    if tuple(value.shape) != tuple(lead) + (dim,):
        raise ValueError(f"value must have shape {tuple(lead) + (dim,)}")
    if tuple(weight.shape) != (hc_mult, dim):
        raise ValueError(f"weight must have shape {(hc_mult, dim)}")
    if x.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        raise TypeError("x and key must be bfloat16")
    if value.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise TypeError("value must be bfloat16 and weight must be float32")
    tensors = (x, key, value, weight)
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("x, key, value and weight must be contiguous")
    if any(tensor.device != x.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
    if image_mask is not None:
        if tuple(image_mask.shape) != tuple(lead) or image_mask.dtype != torch.bool:
            raise ValueError(f"image_mask must be bool with shape {tuple(lead)}")
        if image_mask.device != x.device:
            raise ValueError("image_mask must be on the same device as x")
        if not image_mask.is_contiguous():
            raise ValueError("image_mask must be contiguous")

    out = torch.empty_like(x)
    if x.ndim == 3:
        x3, key3, value2, out3, mask1 = x, key, value, out, image_mask
    elif x.ndim == 2:
        x3 = x.view(total_tokens, 1, dim)
        key3 = key.view(total_tokens, 1, dim)
        value2 = value
        out3 = out.view(total_tokens, 1, dim)
        mask1 = image_mask
    else:
        x3 = x.flatten(0, -3)
        key3 = key.flatten(0, -3)
        value2 = value.flatten(0, -2)
        out3 = out.flatten(0, -3)
        mask1 = image_mask.flatten() if image_mask is not None else None
    if mask1 is not None:
        EngramGate(eps, clamp_value, dim, hc_mult).run(
            x3, key3, value2, weight, out3, mask=mask1
        )
        return out
    EngramGate(eps, clamp_value, dim, hc_mult).run(x3, key3, value2, weight, out3)
    return out
