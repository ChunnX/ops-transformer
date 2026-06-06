#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import math
import copy
import numpy as np
import random

COLOR_YELLOW = "\033[33m"
YELLOW_RESET = "\033[0m"


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)


FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORM = 0.015625

_HAS_TORCH_FP8 = hasattr(torch, 'float8_e4m3fn')

if _HAS_TORCH_FP8:
    _fp8_dtype = torch.float8_e4m3fn
else:
    _fp8_dtype = None


def dequant_int8(x_int8, scale):
    return x_int8.float() * scale


def quant_fp8_e4m3_block(x_float, block_scale_e8m0, block_size=32):
    x_int = torch.clamp(torch.round(x_float / block_scale_e8m0), -448, 448).to(torch.int8)
    return x_int


def dequant_fp8_e4m3_block(x_int, block_scale_e8m0, block_size=32):
    return x_int.float() * block_scale_e8m0


def quant_fp8_e4m3_per_channel(x_float, scale):
    return torch.clamp(torch.round(x_float / scale), -448, 448).to(torch.int8)


def dequant_fp8_e4m3_per_channel(x_int, scale):
    return x_int.float() * scale


def dynamic_quant(x_float, quant_max=127):
    abs_max = torch.max(torch.abs(x_float), dim=-1, keepdim=True)[0]
    scale = torch.clamp(abs_max / float(quant_max), min=1e-10)
    x_quant = torch.clamp(torch.round(x_float / scale), -quant_max, quant_max).to(torch.int8)
    return x_quant, scale.squeeze(-1)


def dynamic_quant_per_tile(x_float, tile_size, clip_alpha=None):
    if clip_alpha is not None:
        x_float = torch.clamp(x_float, -clip_alpha, clip_alpha)
    rows = x_float.shape[0]
    cols = x_float.shape[1]
    num_tiles = (rows + tile_size - 1) // tile_size
    x_quant = torch.zeros(rows, cols, dtype=torch.int8)
    scales = torch.zeros(rows, dtype=torch.float32)
    for t in range(num_tiles):
        start = t * tile_size
        end = min(start + tile_size, rows)
        tile = x_float[start:end]
        tile_abs_max = torch.max(torch.abs(tile.reshape(-1)))
        tile_scale = max(tile_abs_max / 127.0, 1e-10)
        x_quant[start:end] = torch.clamp(torch.round(tile / tile_scale), -128, 127).to(torch.int8)
        scales[start:end] = tile_scale
    return x_quant, scales


def generate_fp8_e8m0_scale(shape, block_size=32):
    exponents = torch.randint(0, 254, shape, dtype=torch.int32)
    return torch.pow(2.0, (exponents.float() - 127.0))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GeneralizedPrologV3:
    """
    Golden reference for MlaPrologV3.

    All computation is done in float32.
    Quantized inputs/weights are dequantized to float before matmul.
    Outputs are produced in the same dtype as the kernel.

    weight_quant_mode: 0=NO_QUANT, 1=PARTIAL, 2=FULL, 3=MXFP8, 4=FP8, 5=HIF8
    kv_cache_quant_mode: 0=NO_QUANT, 1=PER_TENSOR, 2=PER_CHANNEL, 3=PER_TILE
    query_quant_mode: 0=NO_QUANT, 1=QUANT
    """

    def __init__(self, params):
        self.batch_size, self.He, self.Hcq, self.Hckv, self.q_head_num, self.kv_head_num, \
            self.head_dim, self.rope_head_dim, \
            self.q_seq, self.kv_seq, self.block_size, self.input_layout, \
            self.cache_mode, self.cq_epsilon, self.ckv_epsilon, self.dtype, \
            self.weight_quant_mode, self.kv_cache_quant_mode, self.query_quant_mode, \
            self.ckvkr_repo_mode, self.quant_scale_repo_mode, self.tile_size, \
            self.qc_qr_scale, self.kc_scale, self.query_norm_flag = params

        self.is_no_quant = (self.weight_quant_mode == 0)
        self.is_partial = (self.weight_quant_mode == 1)
        self.is_full = (self.weight_quant_mode == 2)
        self.is_mxfp8 = (self.weight_quant_mode == 3)
        self.is_fp8 = (self.weight_quant_mode == 4)
        self.is_hif8 = (self.weight_quant_mode == 5)
        self.is_kv_no_quant = (self.kv_cache_quant_mode == 0)
        self.is_kv_per_tensor = (self.kv_cache_quant_mode == 1)
        self.is_kv_per_channel = (self.kv_cache_quant_mode == 2)
        self.is_kv_per_tile = (self.kv_cache_quant_mode == 3)

        self._need_int8_query = (
            (self.is_partial and not self.is_kv_no_quant) or
            (self.is_full and self.is_kv_per_tensor) or
            (self.is_mxfp8 and self.is_kv_per_tensor) or
            (self.is_fp8 and self.is_kv_per_tensor) or
            (self.is_hif8 and self.is_kv_per_tensor) or
            (self.query_quant_mode == 1)
        )
        # dequantScaleQNope only produced for full-quant + kv_per_tensor
        self._need_dequant_scale_qnope = (
            (self.is_full and self.is_kv_per_tensor) or
            (self.is_mxfp8 and self.is_kv_per_tensor) or
            (self.is_fp8 and self.is_kv_per_tensor) or
            (self.is_hif8 and self.is_kv_per_tensor)
        )
        self._need_int8_kv = not self.is_kv_no_quant

    def _dequant_token_x(self, token_x, dequant_scale_x):
        if self.is_full:
            return dequant_int8(token_x, dequant_scale_x)
        if self.is_mxfp8 or self.is_fp8 or self.is_hif8:
            return self._dequant_intlike(token_x, dequant_scale_x)
        return token_x.float()

    def _dequant_weight(self, weight, dequant_scale):
        if self.is_partial or self.is_full:
            return dequant_int8(weight, dequant_scale)
        if self.is_mxfp8 or self.is_fp8 or self.is_hif8:
            return self._dequant_intlike(weight, dequant_scale)
        return weight.float()

    def _dequant_intlike(self, x_int, scale):
        """Dequant for FP8E4M3/HIF8: these are stored as int8/uint8 on CPU but
        represent FP8 values. For golden, we treat them as int8 with scale.
        MXFP8 block scale needs special handling (32-elements per scale)."""
        if self.is_mxfp8 and scale is not None and scale.dim() > 1:
            return dequant_fp8_e4m3_block(x_int, scale)
        return x_int.float() * scale

    def _smooth_scale_cq(self, x_float, smooth_scale):
        if (self.is_partial or self.is_full) and smooth_scale is not None:
            return x_float * smooth_scale
        return x_float

    def _quant_kv_cache_out(self, x_float, clip_alpha=None):
        if self.is_kv_no_quant:
            return x_float.to(torch.bfloat16)
        if self.is_kv_per_tile:
            return dynamic_quant_per_tile(x_float, self.tile_size, clip_alpha)[0]
        return dynamic_quant(x_float)[0]

    def _quant_kv_cache_out_with_scales(self, x_float, clip_alpha=None):
        if self.is_kv_per_tile:
            q, scales = dynamic_quant_per_tile(x_float, self.tile_size, clip_alpha)
            if self.quant_scale_repo_mode == 1:
                return q, scales
            return q, None
        if self.is_kv_no_quant:
            return x_float.to(torch.bfloat16), None
        return dynamic_quant(x_float)[0], None

    def _get_kv_cache_cols(self):
        cols = self.Hckv
        if self.is_kv_per_tile and self.quant_scale_repo_mode == 1:
            if self.ckvkr_repo_mode == 1:
                cols += self.rope_head_dim * 2
            tile_num = self.Hckv // self.tile_size
            cols += tile_num * 4
        return cols

    def _quant_query_out(self, x_float):
        if self._need_int8_query:
            return dynamic_quant(x_float)
        return x_float.to(torch.bfloat16), None

    def forward(self, vars_list):
        B = self.batch_size
        S1 = self.q_seq
        S2 = self.kv_seq
        D = self.head_dim
        Dr = self.rope_head_dim
        N1 = self.q_head_num
        N2 = self.kv_head_num
        He = self.He
        Hckv = self.Hckv
        Hcq = self.Hcq
        BlockSize = self.block_size
        T = B * S1
        block_num = math.ceil(B * S2 / BlockSize)

        (token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
         sin, cos, index_table, kv_cache, kr_cache,
         dequant_scale_x, dequant_scale_wdq, dequant_scale_wuqqr, dequant_scale_wdkvkr,
         kv_quant_scale_ckv, kv_quant_scale_ckr,
         smooth_scale_cq, actual_seq_len, nope_clip_alpha) = [t.cpu() for t in vars_list]

        print("[INFO]================ Golden (wq=%d kvq=%d qq=%d) ================" %
              (self.weight_quant_mode, self.kv_cache_quant_mode, self.query_quant_mode))

        # ==== Stage 1: MatmulCq (tokenX * weightDq) ====
        token_x_f = self._dequant_token_x(token_x, dequant_scale_x).reshape(T, He)
        w_dq_f = self._dequant_weight(w_dq, dequant_scale_wdq)
        cq_float = torch.matmul(token_x_f, w_dq_f)
        print("[INFO]MatmulCq: " + str(cq_float.shape))

        # ==== Stage 2: RmsNormCq ====
        ep1 = float(self.cq_epsilon)
        norm_cq_float = cq_float / torch.sqrt(torch.mean(cq_float ** 2, dim=-1, keepdim=True) + ep1)
        norm_cq_float = norm_cq_float * gamma_cq.float()
        norm_cq_float = self._smooth_scale_cq(norm_cq_float, smooth_scale_cq)

        # Quantize rmsnorm output for quantized modes
        _query_norm_out = None
        _dequant_scale_qnorm_out = None
        if self.is_no_quant:
            norm_cq_int8 = norm_cq_float
            dequant_scale_cq_out = None
            if self.query_norm_flag:
                _query_norm_out = norm_cq_float.to(torch.bfloat16)
        else:
            if self.is_mxfp8 or self.is_fp8:
                qmax = 448
            else:
                qmax = 127
            norm_cq_int8, dequant_scale_cq_out = dynamic_quant(norm_cq_float, quant_max=qmax)
            if self.query_norm_flag:
                _query_norm_out = norm_cq_int8
                _dequant_scale_qnorm_out = dequant_scale_cq_out
        print("[INFO]RmsNormCq: " + str(norm_cq_int8.shape))

        # ==== Stage 3: MatmulQcQr ====
        w_uq_qr_f = self._dequant_weight(w_uq_qr, dequant_scale_wuqqr)
        qcqr_float = torch.matmul(norm_cq_int8.float(), w_uq_qr_f).reshape(T, N1, D + Dr)
        print("[INFO]MatmulQcQr: " + str(qcqr_float.shape))

        # ==== Stage 4: Split Qc / Qr ====
        qc_float = qcqr_float[:, :, :D]
        qr_float = qcqr_float[:, :, D:]

        # Apply qc_qr_scale to Qc (query correction factor)
        if self.qc_qr_scale != 1.0:
            qc_float = qc_float * self.qc_qr_scale

        # ==== Stage 5: MatmulQn (Qc * weightUk) ====
        qn_float = torch.zeros(T, Hckv)
        for n1_idx in range(N1):
            qn_float += torch.matmul(qc_float[:, n1_idx], w_uk[n1_idx].float())
        qn_quant, dequant_scale_qnope = self._quant_query_out(qn_float)
        out1 = qn_quant.reshape(B, S1, N1, Hckv)
        print("[INFO]MatmulQn: " + str(out1.shape))

        # ==== Stage 6: RopeQr ====
        cos_f = cos.reshape(T, 1, Dr).float()
        sin_f = sin.reshape(T, 1, Dr).float()
        q = qr_float.reshape(T, N1, Dr // 2, 2).transpose(2, 3).reshape(T, N1, Dr)
        q_rot = q * cos_f.repeat(1, N1, 1) + rotate_half(q) * sin_f.repeat(1, N1, 1)
        out2 = q_rot.to(torch.bfloat16).reshape(B, S1, N1, Dr)
        print("[INFO]RopeQr: " + str(out2.shape))

        # ==== Stage 7: MatmulCkvKr ====
        w_dkvkr_f = self._dequant_weight(w_dkv_kr, dequant_scale_wdkvkr)
        ckvkr_float = torch.matmul(token_x_f, w_dkvkr_f)
        ckv_float = ckvkr_float[:, :Hckv]
        kr_float = ckvkr_float[:, Hckv:]
        print("[INFO]MatmulCkvKr: " + str(ckv_float.shape))

        # ==== Stage 8: RmsNormCkv ====
        ep2 = float(self.ckv_epsilon)
        norm_ckv_float = ckv_float / torch.sqrt(torch.mean(ckv_float ** 2, dim=-1, keepdim=True) + ep2)
        norm_ckv_float = norm_ckv_float * gamma_ckv.float()

        # ==== Stage 9: Scatter Ckv / Rope+Scatter Kr ====
        if kv_cache.numel() == 0:
            return (out1.to(torch.bfloat16), out2.to(torch.bfloat16), kv_cache, kr_cache,
                    dequant_scale_qnope, _query_norm_out, _dequant_scale_qnorm_out)

        clip_val = nope_clip_alpha.item() if nope_clip_alpha is not None else None
        kv_cpu = copy.deepcopy(kv_cache)
        kv_orig = kv_cpu.shape
        kv_cache_cols = self._get_kv_cache_cols()
        if self.cache_mode == "PA_BSND":
            kv_cpu = kv_cpu.reshape(kv_orig[0] * BlockSize, N2, kv_cache_cols)
        elif self.cache_mode in ("BSND", "TND"):
            kv_cpu = kv_cpu.reshape(B * S2, N2, kv_cache_cols)

        kr_cpu = copy.deepcopy(kr_cache)
        kr_orig = kr_cpu.shape
        if self.cache_mode == "PA_BSND":
            kr_cpu = kr_cpu.reshape(kr_orig[0] * BlockSize, N2, Dr)
        elif self.cache_mode in ("BSND", "TND"):
            kr_cpu = kr_cpu.reshape(B * S2, N2, Dr)

        # Apply kc_scale to Ckv (KV scale correction factor)
        if self.kc_scale != 1.0:
            norm_ckv_float = norm_ckv_float * self.kc_scale

        ckv_src, per_tile_scales = self._quant_kv_cache_out_with_scales(norm_ckv_float, clip_val)

        k = kr_float.reshape(T, 1, Dr // 2, 2).transpose(2, 3).reshape(T, Dr)
        kr_rope = (k * cos.reshape(T, Dr).float() +
                   rotate_half(k) * sin.reshape(T, Dr).float())

        for i in range(T):
            idx = int(index_table[i].item())
            for j in range(N2):
                kv_cpu[idx, j, :Hckv] = ckv_src[i].to(kv_cpu.dtype)
                if per_tile_scales is not None:
                    tile_num = Hckv // self.tile_size
                    scale_bytes = per_tile_scales[i * tile_num:(i + 1) * tile_num].numpy().tobytes()
                    scale_tensor = torch.frombuffer(bytearray(scale_bytes), dtype=torch.int8)
                    kv_cpu[idx, j, Hckv:Hckv + len(scale_tensor)] = scale_tensor
                kr_cpu[idx, j, :] = kr_rope[i].to(kr_cpu.dtype)

        out3 = kv_cpu.reshape(kv_orig)
        out4 = kr_cpu.reshape(kr_orig)
        print("[INFO]Scatter: kv=%s kr=%s" % (str(out3.shape), str(out4.shape)))
        print("[INFO]================ Golden success (wq=%d) ================" % self.weight_quant_mode)

        return (out1.to(torch.bfloat16), out2.to(torch.bfloat16), out3, out4,
                dequant_scale_qnope, _query_norm_out, _dequant_scale_qnorm_out)


# ============================================================
# Test entry points
# ============================================================

def _get_kv_cache_size(Hckv, Dr, kv_cache_quant_mode, quant_scale_repo_mode, ckvkr_repo_mode, tile_size):
    cols = Hckv
    if kv_cache_quant_mode == 3 and quant_scale_repo_mode == 1:
        if ckvkr_repo_mode == 1:
            cols += Dr * 2  # Kr cache stored inline before scales
        tile_num = Hckv // tile_size
        cols += tile_num * 4
    return cols


def test_prologv3_no_quant(params):
    """weight_quant_mode=0, kv_cache_quant_mode=0"""
    batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
        q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype, \
        weight_quant_mode, kv_cache_quant_mode, query_quant_mode, \
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query_norm_flag = params

    block_num = math.ceil(batch_size * kv_seq / block_size)
    seed = 3
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    import torch_npu

    token_x = torch.rand(batch_size, q_seq, He, dtype=torch.bfloat16, generator=g).npu()
    w_dq_nz = torch_npu.npu_format_cast(torch.rand(He, Hcq, dtype=torch.bfloat16, generator=g).npu().contiguous(), 29)
    w_uq_qr_nz = torch_npu.npu_format_cast(
        torch.rand(Hcq, q_head_num * (head_dim + rope_head_dim), dtype=torch.bfloat16, generator=g).npu().contiguous(), 29)
    w_uk = torch.rand(q_head_num, head_dim, Hckv, dtype=torch.bfloat16, generator=g).npu()
    w_dkv_kr_nz = torch_npu.npu_format_cast(
        torch.rand(He, Hckv + rope_head_dim, dtype=torch.bfloat16, generator=g).npu().contiguous(), 29)
    gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16, generator=g).npu()
    gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16, generator=g).npu()
    rope_sin = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()
    rope_cos = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()

    total_el = batch_size * kv_seq
    arr = np.arange(total_el)
    np.random.shuffle(arr)
    cache_idx = torch.from_numpy(arr[:batch_size * q_seq].reshape((batch_size, q_seq))).npu()
    kv_cache = torch.rand(block_num, block_size, kv_head_num, Hckv, dtype=torch.bfloat16, generator=g).npu()
    kr_cache = torch.rand(block_num, block_size, kv_head_num, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()

    vars_list = [token_x.cpu(), w_dq_nz.cpu(), w_uq_qr_nz.cpu(), w_uk.cpu(), w_dkv_kr_nz.cpu(),
                 gamma_cq.cpu(), gamma_ckv.cpu(), rope_sin.cpu(), rope_cos.cpu(), cache_idx.cpu(),
                 kv_cache.cpu(), kr_cache.cpu(),
                 None, None, None, None, None, None, None, None, None]
    golden = GeneralizedPrologV3(params)
    expect = golden.forward(vars_list)

    result = torch_npu.npu_mla_prolog_v3(
        token_x, w_dq_nz, w_uq_qr_nz, w_uk, w_dkv_kr_nz, gamma_cq, gamma_ckv,
        rope_sin, rope_cos, kv_cache, kr_cache,
        cache_index=cache_idx,
        rmsnorm_epsilon_cq=cq_epsilon, rmsnorm_epsilon_ckv=ckv_epsilon,
        cache_mode=cache_mode,
        weight_quant_mode=weight_quant_mode, kv_cache_quant_mode=kv_cache_quant_mode,
        query_quant_mode=query_quant_mode, ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode, tile_size=tile_size,
        qc_qr_scale=qc_qr_scale, kc_scale=kc_scale, query_norm_flag=query_norm_flag)
    torch.npu.synchronize()
    return expect, result


def test_prologv3_partial_quant(params):
    """weight_quant_mode=1"""
    batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
        q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype, \
        weight_quant_mode, kv_cache_quant_mode, query_quant_mode, \
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query_norm_flag = params

    block_num = math.ceil(batch_size * kv_seq / block_size)
    seed = 3
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    import torch_npu

    token_x = torch.rand(batch_size, q_seq, He, dtype=torch.bfloat16, generator=g).npu()
    w_dq_nz = torch_npu.npu_format_cast(torch.rand(He, Hcq, dtype=torch.bfloat16, generator=g).npu().contiguous(), 29)

    # weightUqQr: int8 NZ
    w_uq_qr_f = torch.rand(Hcq, q_head_num * (head_dim + rope_head_dim), dtype=torch.float32, generator=g)
    dswuqqr = torch.rand(Hcq, q_head_num * (head_dim + rope_head_dim), dtype=torch.float32, generator=g) * 0.01 + 0.001
    w_uq_qr_i8 = torch.clamp(torch.round(w_uq_qr_f / dswuqqr), -128, 127).to(torch.int8)
    w_uq_qr_nz = torch_npu.npu_format_cast(w_uq_qr_i8.npu().contiguous(), 29)

    w_uk = torch.rand(q_head_num, head_dim, Hckv, dtype=torch.bfloat16, generator=g).npu()
    w_dkv_kr_nz = torch_npu.npu_format_cast(
        torch.rand(He, Hckv + rope_head_dim, dtype=torch.bfloat16, generator=g).npu().contiguous(), 29)
    gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16, generator=g).npu()
    gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16, generator=g).npu()
    rope_sin = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()
    rope_cos = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()

    total_el = batch_size * kv_seq
    arr = np.arange(total_el)
    np.random.shuffle(arr)
    cache_idx = torch.from_numpy(arr[:batch_size * q_seq].reshape((batch_size, q_seq))).npu()

    # Smooth scale
    smooth_cq = torch.rand(Hcq, dtype=torch.float32, generator=g).cpu()

    kv_cols = _get_kv_cache_size(Hckv, rope_head_dim, kv_cache_quant_mode, quant_scale_repo_mode, ckvkr_repo_mode, tile_size)
    kv_dtype = torch.int8 if kv_cache_quant_mode != 0 else torch.bfloat16
    kv_cache = torch.rand(block_num, block_size, kv_head_num, kv_cols, dtype=kv_dtype, generator=g).npu()
    kr_cache = torch.rand(block_num, block_size, kv_head_num, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()

    clip_val = torch.tensor(2.0, dtype=torch.float32) if kv_cache_quant_mode == 3 else None

    kv_s = None
    if kv_cache_quant_mode == 2:
        kv_s = torch.rand(Hckv, dtype=torch.float32, generator=g).cpu()

    vars_list = [token_x.cpu(), w_dq_nz.cpu(), w_uq_qr_i8, w_uk.cpu(), w_dkv_kr_nz.cpu(),
                 gamma_cq.cpu(), gamma_ckv.cpu(), rope_sin.cpu(), rope_cos.cpu(), cache_idx.cpu(),
                 kv_cache.cpu(), kr_cache.cpu(),
                 None, None, dswuqqr, None, kv_s, None, smooth_cq.cpu(), None, clip_val]
    golden = GeneralizedPrologV3(params)
    expect = golden.forward(vars_list)

    result = torch_npu.npu_mla_prolog_v3(
        token_x, w_dq_nz, w_uq_qr_nz, w_uk, w_dkv_kr_nz, gamma_cq, gamma_ckv,
        rope_sin, rope_cos, kv_cache, kr_cache,
        cache_index=cache_idx,
        rmsnorm_epsilon_cq=cq_epsilon, rmsnorm_epsilon_ckv=ckv_epsilon,
        cache_mode=cache_mode,
        weight_quant_mode=weight_quant_mode, kv_cache_quant_mode=kv_cache_quant_mode,
        query_quant_mode=query_quant_mode, ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode, tile_size=tile_size,
        qc_qr_scale=qc_qr_scale, kc_scale=kc_scale, query_norm_flag=query_norm_flag,
        dequant_scale_wuq_qr=dswuqqr.npu() if dswuqqr is not None else None,
        smooth_scale_cq=smooth_cq.npu(),
    )
    torch.npu.synchronize()
    return expect, result


def test_prologv3_full_quant(params):
    """weight_quant_mode=2"""
    batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
        q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype, \
        weight_quant_mode, kv_cache_quant_mode, query_quant_mode, \
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query_norm_flag = params

    block_num = math.ceil(batch_size * kv_seq / block_size)
    T = batch_size * q_seq
    seed = 3
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    import torch_npu

    # TokenX: int8
    dsx = torch.rand(T, dtype=torch.float32, generator=g) * 0.01 + 0.001
    tok_f = torch.rand(batch_size, q_seq, He, dtype=torch.float32, generator=g)
    token_x = torch.clamp(torch.round(tok_f / dsx.reshape(T, 1, 1)), -128, 127).to(torch.int8).npu()

    # WeightDq: int8 NZ
    dswdq = torch.rand(He, Hcq, dtype=torch.float32, generator=g) * 0.01 + 0.001
    w_dq_i8 = torch.clamp(torch.round(torch.rand(He, Hcq, torch.float32, generator=g) / dswdq), -128, 127).to(torch.int8)
    w_dq_nz = torch_npu.npu_format_cast(w_dq_i8.npu().contiguous(), 29)

    # WeightUqQr: int8 NZ
    dswuqqr = torch.rand(Hcq, q_head_num * (head_dim + rope_head_dim), torch.float32, generator=g) * 0.01 + 0.001
    w_uq_qr_i8 = torch.clamp(
        torch.round(torch.rand(Hcq, q_head_num * (head_dim + rope_head_dim), torch.float32, generator=g) / dswuqqr),
        -128, 127).to(torch.int8)
    w_uq_qr_nz = torch_npu.npu_format_cast(w_uq_qr_i8.npu().contiguous(), 29)

    w_uk = torch.rand(q_head_num, head_dim, Hckv, dtype=torch.float32, generator=g).npu()

    # WeightDkvKr: int8 NZ
    dswdkvkr = torch.rand(He, Hckv + rope_head_dim, torch.float32, generator=g) * 0.01 + 0.001
    w_dkv_kr_i8 = torch.clamp(
        torch.round(torch.rand(He, Hckv + rope_head_dim, torch.float32, generator=g) / dswdkvkr), -128, 127).to(torch.int8)
    w_dkv_kr_nz = torch_npu.npu_format_cast(w_dkv_kr_i8.npu().contiguous(), 29)

    gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16, generator=g).npu()
    gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16, generator=g).npu()
    rope_sin = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()
    rope_cos = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()

    total_el = batch_size * kv_seq
    arr = np.arange(total_el)
    np.random.shuffle(arr)
    cache_idx = torch.from_numpy(arr[:T].reshape((batch_size, q_seq))).npu()

    smooth_cq = torch.rand(Hcq, dtype=torch.float32, generator=g).cpu()

    kv_cols = _get_kv_cache_size(Hckv, rope_head_dim, kv_cache_quant_mode, quant_scale_repo_mode, ckvkr_repo_mode, tile_size)
    kv_dtype = torch.int8 if kv_cache_quant_mode != 0 else torch.bfloat16
    kv_cache = torch.rand(block_num, block_size, kv_head_num, kv_cols, dtype=kv_dtype, generator=g).npu()
    kr_cache = torch.rand(block_num, block_size, kv_head_num, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()

    clip_val = torch.tensor(2.0, dtype=torch.float32) if kv_cache_quant_mode == 3 else None
    kv_s = torch.rand(Hckv, dtype=torch.float32, generator=g).cpu() if kv_cache_quant_mode == 2 else None

    vars_list = [token_x.cpu(), w_dq_i8, w_uq_qr_i8, w_uk.cpu(), w_dkv_kr_i8,
                 gamma_cq.cpu(), gamma_ckv.cpu(), rope_sin.cpu(), rope_cos.cpu(), cache_idx.cpu(),
                 kv_cache.cpu(), kr_cache.cpu(),
                 dsx, dswdq, dswuqqr, dswdkvkr, kv_s, None, smooth_cq.cpu(), None, clip_val]
    golden = GeneralizedPrologV3(params)
    expect = golden.forward(vars_list)

    result = torch_npu.npu_mla_prolog_v3(
        token_x, w_dq_nz, w_uq_qr_nz, w_uk, w_dkv_kr_nz, gamma_cq, gamma_ckv,
        rope_sin, rope_cos, kv_cache, kr_cache,
        cache_index=cache_idx,
        rmsnorm_epsilon_cq=cq_epsilon, rmsnorm_epsilon_ckv=ckv_epsilon,
        cache_mode=cache_mode,
        weight_quant_mode=weight_quant_mode, kv_cache_quant_mode=kv_cache_quant_mode,
        query_quant_mode=query_quant_mode, ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode, tile_size=tile_size,
        qc_qr_scale=qc_qr_scale, kc_scale=kc_scale, query_norm_flag=query_norm_flag,
        dequant_scale_x=dsx.reshape(T, 1).npu(),
        dequant_scale_wdq=dswdq.npu(),
        dequant_scale_wuq_qr=dswuqqr.npu(),
        dequant_scale_wdkv_kr=dswdkvkr.npu(),
        smooth_scale_cq=smooth_cq.npu(),
    )
    torch.npu.synchronize()
    return expect, result


def test_prologv3_mxfp8(params):
    """weight_quant_mode=3 (MXFP8: FP8E4M3 + FP8E8M0 block scale)"""
    batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
        q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype, \
        weight_quant_mode, kv_cache_quant_mode, query_quant_mode, \
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query_norm_flag = params

    block_num = math.ceil(batch_size * kv_seq / block_size)
    T = batch_size * q_seq
    seed = 3
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    import torch_npu

    _bs = 32

    def _make_fp8_e4m3_data(f_shape, block_scale_shape, f_scale_shape):
        raw_f = torch.rand(f_shape, dtype=torch.float32, generator=g)
        block_s = generate_fp8_e8m0_scale(block_scale_shape)
        per_ch_s = torch.rand(f_scale_shape, dtype=torch.float32, generator=g) * 0.01 + 0.01
        i8 = quant_fp8_e4m3_block(raw_f, _expand_block_scale(block_s, f_shape))
        return i8, block_s, per_ch_s

    def _expand_block_scale(block_s, target_shape):
        if len(target_shape) == 2:
            H, W = target_shape
            num_blocks = W // _bs
            return block_s[:, :num_blocks].repeat_interleave(_bs, dim=1)[:, :W]
        return block_s

    # TokenX: int8 (simulating FP8E4M3)
    tok_f = torch.rand(batch_size, q_seq, He, dtype=torch.float32, generator=g)
    dsx = torch.rand(T, dtype=torch.float32, generator=g) * 0.01 + 0.001
    token_x_i8 = torch.clamp(torch.round(tok_f / dsx.reshape(T, 1, 1)), -128, 127).to(torch.int8).npu()

    # Weight Dq
    blk_dq = generate_fp8_e8m0_scale((He, (Hcq + _bs - 1) // _bs))
    dswdq = torch.rand(He, Hcq, dtype=torch.float32, generator=g) * 0.01 + 0.01
    w_dq_f = torch.rand(He, Hcq, dtype=torch.float32, generator=g)
    w_dq_i8 = quant_fp8_e4m3_block(w_dq_f, _expand_block_scale(blk_dq, (He, Hcq)))
    w_dq_nz = torch_npu.npu_format_cast(w_dq_i8.npu().contiguous(), 29)

    # Weight UqQr
    wuq_dim = q_head_num * (head_dim + rope_head_dim)
    blk_uq = generate_fp8_e8m0_scale((Hcq, (wuq_dim + _bs - 1) // _bs))
    dswuqqr = torch.rand(Hcq, wuq_dim, dtype=torch.float32, generator=g) * 0.01 + 0.01
    w_uq_qr_f = torch.rand(Hcq, wuq_dim, dtype=torch.float32, generator=g)
    w_uq_qr_i8 = quant_fp8_e4m3_block(w_uq_qr_f, _expand_block_scale(blk_uq, (Hcq, wuq_dim)))
    w_uq_qr_nz = torch_npu.npu_format_cast(w_uq_qr_i8.npu().contiguous(), 29)

    w_uk = torch.rand(q_head_num, head_dim, Hckv, dtype=torch.float32, generator=g).npu()

    # Weight DkvKr
    dkv_dim = Hckv + rope_head_dim
    blk_dkv = generate_fp8_e8m0_scale((He, (dkv_dim + _bs - 1) // _bs))
    dswdkvkr = torch.rand(He, dkv_dim, dtype=torch.float32, generator=g) * 0.01 + 0.01
    w_dkv_kr_f = torch.rand(He, dkv_dim, dtype=torch.float32, generator=g)
    w_dkv_kr_i8 = quant_fp8_e4m3_block(w_dkv_kr_f, _expand_block_scale(blk_dkv, (He, dkv_dim)))
    w_dkv_kr_nz = torch_npu.npu_format_cast(w_dkv_kr_i8.npu().contiguous(), 29)

    gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16, generator=g).npu()
    gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16, generator=g).npu()
    rope_sin = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()
    rope_cos = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()

    total_el = batch_size * kv_seq
    arr = np.arange(total_el)
    np.random.shuffle(arr)
    cache_idx = torch.from_numpy(arr[:T].reshape((batch_size, q_seq))).npu()

    smooth_cq = torch.rand(Hcq, dtype=torch.float32, generator=g).cpu()
    kv_cols = _get_kv_cache_size(Hckv, rope_head_dim, kv_cache_quant_mode, quant_scale_repo_mode, ckvkr_repo_mode, tile_size)
    kv_dtype = torch.int8 if kv_cache_quant_mode != 0 else torch.bfloat16
    kv_cache = torch.rand(block_num, block_size, kv_head_num, kv_cols, dtype=kv_dtype, generator=g).npu()
    kr_cache = torch.rand(block_num, block_size, kv_head_num, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()
    clip_val = torch.tensor(2.0, dtype=torch.float32) if kv_cache_quant_mode == 3 else None

    # Build golden vars (use per-channel float scales for golden dequant)
    vars_list = [token_x_i8, w_dq_i8, w_uq_qr_i8, w_uk.cpu(), w_dkv_kr_i8,
                 gamma_cq.cpu(), gamma_ckv.cpu(), rope_sin.cpu(), rope_cos.cpu(), cache_idx.cpu(),
                 kv_cache.cpu(), kr_cache.cpu(),
                 dsx, dswdq, dswuqqr, dswdkvkr, None, None, smooth_cq.cpu(), None, clip_val]
    golden = GeneralizedPrologV3(params)
    expect = golden.forward(vars_list)

    result = torch_npu.npu_mla_prolog_v3(
        token_x_i8, w_dq_nz, w_uq_qr_nz, w_uk, w_dkv_kr_nz, gamma_cq, gamma_ckv,
        rope_sin, rope_cos, kv_cache, kr_cache,
        cache_index=cache_idx,
        rmsnorm_epsilon_cq=cq_epsilon, rmsnorm_epsilon_ckv=ckv_epsilon,
        cache_mode=cache_mode,
        weight_quant_mode=weight_quant_mode, kv_cache_quant_mode=kv_cache_quant_mode,
        query_quant_mode=query_quant_mode, ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode, tile_size=tile_size,
        qc_qr_scale=qc_qr_scale, kc_scale=kc_scale, query_norm_flag=query_norm_flag,
        dequant_scale_x=dsx.reshape(T, 1).npu(),
        dequant_scale_wdq=dswdq.npu(),
        dequant_scale_wuq_qr=dswuqqr.npu(),
        dequant_scale_wdkv_kr=dswdkvkr.npu(),
        smooth_scale_cq=smooth_cq.npu(),
    )
    torch.npu.synchronize()
    return expect, result


def test_prologv3_fp8(params):
    """weight_quant_mode=4 (FP8: FP8E4M3 + float per-channel scale)"""
    return _test_prologv3_fp8like(params, quant_max=448)


def test_prologv3_hif8(params):
    """weight_quant_mode=5 (HIF8: HIF8 + float per-channel scale)"""
    return _test_prologv3_fp8like(params, quant_max=127)


def _test_prologv3_fp8like(params, quant_max=448):
    """Shared test entry for FP8 (mode 4) and HIF8 (mode 5).
    Both use per-channel float dequant scales like int8 full quant.
    HIF8 uses int8 quant range, FP8 uses FP8E4M3 range."""
    batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
        q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype, \
        weight_quant_mode, kv_cache_quant_mode, query_quant_mode, \
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query_norm_flag = params

    block_num = math.ceil(batch_size * kv_seq / block_size)
    T = batch_size * q_seq
    seed = 3
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    import torch_npu

    dsx = torch.rand(T, dtype=torch.float32, generator=g) * 0.01 + 0.001
    tok_f = torch.rand(batch_size, q_seq, He, dtype=torch.float32, generator=g)
    token_x_i8 = torch.clamp(torch.round(tok_f / dsx.reshape(T, 1, 1)), -quant_max, quant_max).to(torch.int8).npu()

    dswdq = torch.rand(He, Hcq, dtype=torch.float32, generator=g) * 0.001 + 0.001
    w_dq_i8 = torch.clamp(torch.round(torch.rand(He, Hcq, torch.float32, generator=g) / dswdq), -quant_max, quant_max).to(torch.int8)
    w_dq_nz = torch_npu.npu_format_cast(w_dq_i8.npu().contiguous(), 29)

    wuq_dim = q_head_num * (head_dim + rope_head_dim)
    dswuqqr = torch.rand(Hcq, wuq_dim, dtype=torch.float32, generator=g) * 0.001 + 0.001
    w_uq_qr_i8 = torch.clamp(torch.round(torch.rand(Hcq, wuq_dim, torch.float32, generator=g) / dswuqqr), -quant_max, quant_max).to(torch.int8)
    w_uq_qr_nz = torch_npu.npu_format_cast(w_uq_qr_i8.npu().contiguous(), 29)

    w_uk = torch.rand(q_head_num, head_dim, Hckv, dtype=torch.float32, generator=g).npu()

    dkv_dim = Hckv + rope_head_dim
    dswdkvkr = torch.rand(He, dkv_dim, dtype=torch.float32, generator=g) * 0.001 + 0.001
    w_dkv_kr_i8 = torch.clamp(torch.round(torch.rand(He, dkv_dim, torch.float32, generator=g) / dswdkvkr), -quant_max, quant_max).to(torch.int8)
    w_dkv_kr_nz = torch_npu.npu_format_cast(w_dkv_kr_i8.npu().contiguous(), 29)

    gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16, generator=g).npu()
    gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16, generator=g).npu()
    rope_sin = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()
    rope_cos = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()

    total_el = batch_size * kv_seq
    arr = np.arange(total_el)
    np.random.shuffle(arr)
    cache_idx = torch.from_numpy(arr[:T].reshape((batch_size, q_seq))).npu()

    smooth_cq = torch.rand(Hcq, dtype=torch.float32, generator=g).cpu()
    kv_cols = _get_kv_cache_size(Hckv, rope_head_dim, kv_cache_quant_mode, quant_scale_repo_mode, ckvkr_repo_mode, tile_size)
    kv_dtype = torch.int8 if kv_cache_quant_mode != 0 else torch.bfloat16
    kv_cache = torch.rand(block_num, block_size, kv_head_num, kv_cols, dtype=kv_dtype, generator=g).npu()
    kr_cache = torch.rand(block_num, block_size, kv_head_num, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()
    clip_val = torch.tensor(2.0, dtype=torch.float32) if kv_cache_quant_mode == 3 else None

    vars_list = [token_x_i8, w_dq_i8, w_uq_qr_i8, w_uk.cpu(), w_dkv_kr_i8,
                 gamma_cq.cpu(), gamma_ckv.cpu(), rope_sin.cpu(), rope_cos.cpu(), cache_idx.cpu(),
                 kv_cache.cpu(), kr_cache.cpu(),
                 dsx, dswdq, dswuqqr, dswdkvkr, None, None, smooth_cq.cpu(), None, clip_val]
    golden = GeneralizedPrologV3(params)
    expect = golden.forward(vars_list)

    result = torch_npu.npu_mla_prolog_v3(
        token_x_i8, w_dq_nz, w_uq_qr_nz, w_uk, w_dkv_kr_nz, gamma_cq, gamma_ckv,
        rope_sin, rope_cos, kv_cache, kr_cache,
        cache_index=cache_idx,
        rmsnorm_epsilon_cq=cq_epsilon, rmsnorm_epsilon_ckv=ckv_epsilon,
        cache_mode=cache_mode,
        weight_quant_mode=weight_quant_mode, kv_cache_quant_mode=kv_cache_quant_mode,
        query_quant_mode=query_quant_mode, ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode, tile_size=tile_size,
        qc_qr_scale=qc_qr_scale, kc_scale=kc_scale, query_norm_flag=query_norm_flag,
        dequant_scale_x=dsx.reshape(T, 1).npu(),
        dequant_scale_wdq=dswdq.npu(),
        dequant_scale_wuq_qr=dswuqqr.npu(),
        dequant_scale_wdkv_kr=dswdkvkr.npu(),
        smooth_scale_cq=smooth_cq.npu(),
    )
    torch.npu.synchronize()
    return expect, result


TEST_FUNCTIONS = {
    (0, 0): test_prologv3_no_quant,
    (1, 0): test_prologv3_partial_quant,
    (1, 2): test_prologv3_partial_quant,
    (1, 3): test_prologv3_partial_quant,
    (2, 0): test_prologv3_full_quant,
    (2, 1): test_prologv3_full_quant,
    (2, 3): test_prologv3_full_quant,
    (3, 0): test_prologv3_mxfp8,
    (3, 1): test_prologv3_mxfp8,
    (3, 3): test_prologv3_mxfp8,
    (4, 0): test_prologv3_fp8,
    (4, 1): test_prologv3_fp8,
    (4, 3): test_prologv3_fp8,
    (5, 0): test_prologv3_hif8,
    (5, 1): test_prologv3_hif8,
    (5, 3): test_prologv3_hif8,
}
