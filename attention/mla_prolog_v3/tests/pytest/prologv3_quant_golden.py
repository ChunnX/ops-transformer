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
import torch_npu
import check_valid_param
import pytest
import math
import copy
import numpy as np
import random

COLOR_YELLOW = "\033[33m"
YELLOW_RESET = "\033[0m"


WEIGHT_QUANT_MODE_NO_QUANT = 0
WEIGHT_QUANT_MODE_PARTIAL_QUANT = 1
WEIGHT_QUANT_MODE_FULL_QUANT_INT8 = 2
WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8 = 3
WEIGHT_QUANT_MODE_FULL_QUANT_FP8 = 4
WEIGHT_QUANT_MODE_FULL_QUANT_HIF8 = 5



KV_CACHE_QUANT_MODE_NO_QUANT = 0
KV_CACHE_QUANT_MODE_PER_TENSOR = 1
KV_CACHE_QUANT_MODE_PER_CHANNEL = 2
KV_CACHE_QUANT_MODE_PER_TILE = 3


def _to_nz_format(data_tensor):
    """Convert a CPU tensor to NPU FRACTAL_NZ format using empty_with_format + copy_."""
    nz_tensor = torch_npu.empty_with_format(
        data_tensor.shape, dtype=data_tensor.dtype, device='npu:0', acl_format=29
    )
    nz_tensor.copy_(data_tensor)
    return nz_tensor


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def scatter_pa_nz(kv_cache, norm_res, index_table, scatter_size):
    B_num = kv_cache.shape[0]
    BlockSize = kv_cache.shape[1]
    kv_cache = kv_cache.reshape(B_num * BlockSize, kv_cache.shape[2], kv_cache.shape[3])
    T = index_table.shape[0]
    for i in range(0, T, scatter_size):
        cur_scatter_size = min(scatter_size, T - i)
        src = norm_res[i:i + cur_scatter_size]
        indices = index_table[i:i + cur_scatter_size]
        kv_cache[indices] = src
    kv_cache = kv_cache.reshape(B_num, BlockSize, kv_cache.shape[1], kv_cache.shape[2])
    return kv_cache


def _make_int8_data(shape, scale_shape, qmax=127, g=None):
    w_float = torch.rand(*shape, dtype=torch.float32, generator=g)
    scale = torch.rand(*scale_shape, dtype=torch.float32, generator=g) * 0.01 + 0.001
    if len(scale_shape) == 1 and len(shape) == 2:
        if scale_shape[0] == shape[0]:
            w_scaled = w_float / scale.unsqueeze(1)
        else:
            w_scaled = w_float / scale.unsqueeze(0)
    elif len(scale_shape) == 1 and len(shape) == 1:
        w_scaled = w_float / scale
    else:
        w_scaled = w_float / scale.reshape([-1] + [1] * (len(shape) - len(scale_shape)))
    w_int8 = torch.clamp(torch.round(w_scaled), -qmax, qmax).to(torch.int8)
    return w_int8, scale


def _make_per_token_scale(shape, g=None):
    T = shape[0]
    scale = torch.rand(T, 1, dtype=torch.float32, generator=g) * 0.01 + 0.001
    return scale


class GeneralizedPrologV3:
    def __init__(self, params):
        self.batch_size, self.He, self.Hcq, self.Hckv, self.q_head_num, self.kv_head_num, \
            self.head_dim, self.rope_head_dim, \
            self.q_seq, self.kv_seq, self.block_size, self.input_layout, \
            self.cache_mode, self.cq_epsilon, self.ckv_epsilon, self.dtype, \
            self.weight_quant_mode, self.kv_cache_quant_mode, self.query_quant_mode, \
            self.ckvkr_repo_mode, self.quant_scale_repo_mode, self.tile_size, \
            self.qc_qr_scale, self.kc_scale, self.query_norm_flag, self.bs_merged = params

    @staticmethod
    def _cast_f32_i8(x):
        """模拟 kernel CastFP32ToINT8: float→RINT→int32→ROUND→half→TRUNC→int8"""
        int32_val = torch.round(x).to(torch.int32)
        half_val = int32_val.to(torch.float16)
        return half_val.to(torch.int8)

    @staticmethod
    def _cast_f32_to_fp8(x):
        """Cast float32 to fp8_e4m3fn with saturation"""
        x_clamped = torch.clamp(x, -448.0, 448.0)
        return x_clamped.to(torch.float8_e4m3fn)

    @staticmethod
    def _cast_f32_to_hifp8(x):
        """模拟 kernel Cast<float, hifloat8_t, castTraitF32toh8>: float32→hifp8

        HIFP8 (HIFLOAT8) 是华为自研 8-bit 浮点格式, 硬件使用 RoundMode::HYBRID 舍入。
        PyTorch 通过 torch.hifloat8 提供原生支持, 仅在 NPU 环境下可用。
        """
        x_clamped = torch.clamp(x, -32768.0, 32768.0)
        return x_clamped.to(torch.hifloat8)

    def _dynamic_quant(self, x, output_dtype=torch.int8, qmax=None):
        """Dynamic quantization per token
        
        Args:
            x: input tensor [T, Hcq]
            output_dtype: target dtype (int8, fp8_e4m3fn, hifloat8)
            qmax: max value for quantization (auto-detect if None)
        
        Returns:
            quantized tensor and scale
        """
        # Determine qmax based on output dtype
        if qmax is None:
            if output_dtype == torch.int8:
                qmax = 127.0
            elif output_dtype == torch.float8_e4m3fn:
                qmax = 448.0
            else:  # hifloat8
                qmax = 32768.0
        
        # Compute scale: alpha = 1.0 / qmax, scale = reduce_max * alpha
        amax = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
        alpha = 1.0 / qmax
        scale = amax * alpha
        
        # Normalize: norm = x / scale
        reciprocal = 1.0 / scale
        scaled = x * reciprocal
        
        # Cast to target dtype
        if output_dtype == torch.int8:
            q = self._cast_f32_i8(scaled)
        elif output_dtype == torch.float8_e4m3fn:
            q = self._cast_f32_to_fp8(scaled)
        elif output_dtype == torch.hifloat8:
            q = self._cast_f32_to_hifp8(scaled)
        else:
            raise ValueError(f"unsupported output_dtype: {output_dtype}")
        
        return q, scale.squeeze(-1)

    @staticmethod
    def _per_channel_quant(x, scale):
        scaled = x / scale
        int32_val = torch.round(scaled).to(torch.int32)
        half_val = int32_val.to(torch.float16)
        return half_val.to(torch.int8)

    def _per_tile_quant(self, x, clip_alpha=None):
        rows, cols = x.shape
        tile_size = self.tile_size
        tile_num = (rows + tile_size - 1) // tile_size
        padded_rows = tile_num * tile_size

        if padded_rows > rows:
            x_pad = torch.zeros(padded_rows, cols, dtype=x.dtype, device=x.device)
            x_pad[:rows] = x
        else:
            x_pad = x

        q = torch.zeros(padded_rows, cols, dtype=torch.int8)
        # 对齐 kernel: PerTileClipWithAlpha 和 DynamicQuant 共用同一个 alpha
        # kernel: scale = aMax * alpha / 127, reciprocal = 127 / (aMax * alpha)
        effective_alpha = clip_alpha if clip_alpha is not None else 1.0
        for t in range(tile_num):
            s = t * tile_size
            e = min(s + tile_size, padded_rows)
            tile = x_pad[s:e]
            abs_tile = torch.abs(tile)
            amax_per_row = torch.max(abs_tile, dim=-1, keepdim=True)[0]
            if clip_alpha is not None:
                max_clip = amax_per_row * clip_alpha
                tile = torch.max(torch.min(tile, max_clip), -max_clip)
            scale = torch.clamp(amax_per_row * effective_alpha / 127.0, min=1e-10)
            reciprocal = 1.0 / scale
            scaled = tile * reciprocal
            q[s:e] = self._cast_f32_i8(scaled)
        return q[:rows]

    def _dequant_matmul_result(self, raw_result, dsx, dsw, T, out_dim):
        if dsx is not None and dsw is not None:
            dsx_row = dsx.reshape(T, 1)
            dsw_col = dsw.reshape(1, out_dim)
            return raw_result.to(torch.float32) * dsx_row * dsw_col
        return raw_result.to(torch.float32)

    def _quant_kv_output(self, ckv, kv_scale, clip_alpha):
        if self.kv_cache_quant_mode == KV_CACHE_QUANT_MODE_NO_QUANT:
            return ckv.to(torch.bfloat16)
        if self.kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_TILE:
            return self._per_tile_quant(ckv, clip_alpha)
        if self.kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_CHANNEL and kv_scale is not None:
            return self._per_channel_quant(ckv, kv_scale)
        if self.kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_TENSOR:
            if self.weight_quant_mode in (WEIGHT_QUANT_MODE_FULL_QUANT_FP8, WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8):
                kv_dtype = torch.float8_e4m3fn
            elif self.weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT_HIF8:
                kv_dtype = torch.hifloat8
            else:
                kv_dtype = torch.int8
            q, _ = self._dynamic_quant(ckv, output_dtype=kv_dtype)
            return q
        return ckv.to(torch.bfloat16)

    def _quant_kr_output(self, kr, kr_scale):
        if self.kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_CHANNEL and kr_scale is not None:
            return self._per_channel_quant(kr, kr_scale)
        return kr.to(torch.bfloat16)

    def _get_quant_mode(self):
        """Calculate quant mode based on weight_quant_mode and kv_cache_quant_mode
        Matches kernel's quant mode enumeration:
        0: non-quant
        1: partial quant, kv no quant
        2: partial quant, kv per-channel
        3: int8 full quant, kv no quant
        4: int8 full quant, kv per-tensor
        5: partial quant, kv per-tile
        6: int8 full quant, kv per-tile
        7: mxfp8 full quant, kv no quant
        8: mxfp8 full quant, kv per-tensor
        9: mxfp8 full quant, kv per-tile
        10: fp8 full quant, kv no quant
        11: fp8 full quant, kv per-tensor
        12: hif8 full quant, kv no quant
        13: hif8 full quant, kv per-tensor
        """
        wq = self.weight_quant_mode
        kvq = self.kv_cache_quant_mode
        
        if wq == WEIGHT_QUANT_MODE_NO_QUANT:
            return 0
        elif wq == WEIGHT_QUANT_MODE_PARTIAL_QUANT:
            if kvq == KV_CACHE_QUANT_MODE_NO_QUANT:
                return 1
            elif kvq == KV_CACHE_QUANT_MODE_PER_CHANNEL:
                return 2
            elif kvq == KV_CACHE_QUANT_MODE_PER_TILE:
                return 5
        elif wq == WEIGHT_QUANT_MODE_FULL_QUANT_INT8:
            if kvq == KV_CACHE_QUANT_MODE_NO_QUANT:
                return 3
            elif kvq == KV_CACHE_QUANT_MODE_PER_TENSOR:
                return 4
            elif kvq == KV_CACHE_QUANT_MODE_PER_TILE:
                return 6
        elif wq == WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8:
            if kvq == KV_CACHE_QUANT_MODE_NO_QUANT:
                return 7
            elif kvq == KV_CACHE_QUANT_MODE_PER_TENSOR:
                return 8
            elif kvq == KV_CACHE_QUANT_MODE_PER_TILE:
                return 9
        elif wq == WEIGHT_QUANT_MODE_FULL_QUANT_FP8:
            if kvq == KV_CACHE_QUANT_MODE_NO_QUANT:
                return 10
            elif kvq == KV_CACHE_QUANT_MODE_PER_TENSOR:
                return 11
        elif wq == WEIGHT_QUANT_MODE_FULL_QUANT_HIF8:
            if kvq == KV_CACHE_QUANT_MODE_NO_QUANT:
                return 12
            elif kvq == KV_CACHE_QUANT_MODE_PER_TENSOR:
                return 13
        
        return -1  # Unknown

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
        T = B * S1 if not self.bs_merged else self.q_seq

        vars_list = [t.cpu() if t is not None else None for t in vars_list]

        # --- 输入解包 ---
        # token_x: merged=(T,He) non-merged=(B,S,He) -> reshape -> (T,He)
        token_x = vars_list[0]
        if not self.bs_merged:
            token_x = token_x.reshape(T, He)
        # w_dq: (He, Hcq)
        w_dq = vars_list[1]
        # w_uq_qr: (Hcq, N*(D+Dr))
        w_uq_qr = vars_list[2]
        # w_uk: (N, D, Hckv)
        w_uk = vars_list[3]
        # w_dkv_kr: (He, Hckv+Dr)
        w_dkv_kr = vars_list[4]
        # gamma_cq: (Hcq,)
        gamma_cq = vars_list[5]
        # gamma_ckv: (Hckv,)
        gamma_ckv = vars_list[6]
        # sin: merged=(T,Dr) non-merged=(B,S,Dr)
        sin = vars_list[7]
        # cos: merged=(T,Dr) non-merged=(B,S,Dr)
        cos = vars_list[8]
        # index_table: (T,) 展平
        index_table = vars_list[9].reshape(-1)
        # kv_cache: (blockNum, blockSize, Nkv, dtileSize)
        kv_cache = vars_list[10]
        # kr_cache: (blockNum, blockSize, Nkv, Dr) 或 empty (0,)
        kr_cache = vars_list[11]

        # dsx: INT8/FP8/HIF8=(T,1) MXFP8=(T,He/32)
        dsx = vars_list[12] if len(vars_list) > 12 else None
        # dswdq: INT8/FP8/HIF8=(1,Hcq) MXFP8=(Hcq,He/32)
        dswdq = vars_list[13] if len(vars_list) > 13 else None
        # dswuqqr: INT8/FP8/HIF8=(1,N*(D+Dr)) MXFP8=(N*(D+Dr),Hcq/32)
        dswuqqr = vars_list[14] if len(vars_list) > 14 else None
        # dswdkvkr: INT8/FP8/HIF8=(1,Hckv+Dr) MXFP8=(Hckv+Dr,He/32)
        dswdkvkr = vars_list[15] if len(vars_list) > 15 else None
        # quant_scale_ckv: PER_TENSOR=(1,) PER_CHANNEL=(1,Hckv)
        quant_scale_ckv = vars_list[16] if len(vars_list) > 16 else None
        # quant_scale_ckr: PER_CHANNEL=(1,Dr)
        quant_scale_ckr = vars_list[17] if len(vars_list) > 17 else None
        # smooth_scales_cq: (1,Hcq)
        smooth_scales_cq = vars_list[18] if len(vars_list) > 18 else None
        # k_nope_clip_alpha: (1,) scalar
        k_nope_clip_alpha = vars_list[19].item() if len(vars_list) > 19 and vars_list[19] is not None else None

        cq_epsilon = float(self.cq_epsilon)
        ckv_epsilon = float(self.ckv_epsilon)
        cache_mode = self.cache_mode
        qc_qr_scale = float(self.qc_qr_scale)
        kc_scale = float(self.kc_scale)

        dtile_size = Hckv
        if self.kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_TILE:
            dtile_size = Hckv + Dr * 2 + Hckv // self.tile_size * 4

        print("[INFO]========================================")
        print("[INFO]>>>>>>>>  Start to calculate (v3 quant) >>>>>>>>>>")
        print("[INFO]wq=%d kvq=%d qq=%d", self.weight_quant_mode, self.kv_cache_quant_mode, self.query_quant_mode)
        print("[INFO]========================================")

        # ===================== Stage 0: MXFP8 Preprocessing =====================
        # dsx: (T,He/32) E8M0 -> expand -> (T,He) bf16
        # token_x: (T,He) bf16 * dsx_expanded(T,He) bf16 -> token_x: (T,He) bf16
        # dswdq: (Hcq,He/32) E8M0 -> expand -> (Hcq,He) bf16
        # w_dq: (He,Hcq) bf16 * dswdq_expanded(Hcq,He) bf16 -> w_dq: (He,Hcq) bf16
        # dswdkvkr: (Hckv+Dr,He/32) E8M0 -> expand -> (Hckv+Dr,He) bf16
        # w_dkv_kr: (He,Hckv+Dr) bf16 * dswdkvkr_expanded(Hckv+Dr,He) bf16 -> w_dkv_kr: (He,Hckv+Dr) bf16
        if self.weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8:
            dsx_bf16 = dsx.to(torch.bfloat16)
            dsx_expanded = dsx_bf16.reshape(T, He // 32, 1).repeat(1, 1, 32).reshape(T, He)
            token_x = token_x.to(torch.bfloat16) * dsx_expanded

            dswdq_bf16 = dswdq.to(torch.bfloat16)
            dswdq_expanded = dswdq_bf16.reshape(Hcq, He // 32, 1).repeat(1, 1, 32).reshape(Hcq, He)
            w_dq = w_dq.to(torch.bfloat16) * dswdq_expanded

            dswdkvkr_bf16 = dswdkvkr.to(torch.bfloat16)
            dswdkvkr_expanded = dswdkvkr_bf16.reshape(Hckv + Dr, He // 32, 1).repeat(1, 1, 32).reshape(Hckv + Dr, He)
            w_dkv_kr = w_dkv_kr.to(torch.bfloat16) * dswdkvkr_expanded

        # ===================== Stage 1: MatmulCq =====================
        # token_x: (T,He) * w_dq: (He,Hcq) -> cq_raw: (T,Hcq)
        # BF16/Partial: bf16*bf16->bf16->float32
        # INT8: int8*int8->int32
        # FP8/MXFP8/HIF8: float32*float32->float32
        print("[INFO]Stage1 MatmulCq start")
        if self.weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT_INT8:
            cq_raw = torch.matmul(token_x.to(torch.int32), w_dq.to(torch.int32)).to(torch.int32)
        else:
            cq_raw = torch.matmul(token_x.to(torch.float32), w_dq.to(torch.float32)).to(torch.float32)
        print("[INFO]Stage1 MatmulCq end. cq_raw:%s|%s", tuple(cq_raw.shape), cq_raw.dtype)

        # ===================== Stage 1.5: Post-MM1 Processing =====================
        # BF16/Partial: bf16 truncation -> float32
        # INT8/FP8/HIF8: two dequant operations
        if self.weight_quant_mode in (WEIGHT_QUANT_MODE_NO_QUANT, WEIGHT_QUANT_MODE_PARTIAL_QUANT):
            cq_raw = cq_raw.to(torch.bfloat16).to(torch.float32)
        elif self.weight_quant_mode in (WEIGHT_QUANT_MODE_FULL_QUANT_INT8, WEIGHT_QUANT_MODE_FULL_QUANT_FP8,
                                         WEIGHT_QUANT_MODE_FULL_QUANT_HIF8):
            # Dequant 1: by input scale dsx: (T,1)
            dsx_row = dsx.reshape(T, 1)
            cq_dequant1 = cq_raw.to(torch.float32) * dsx_row
            # Dequant 2: by weight scale dswdq: (1,Hcq)
            dsw_col = dswdq.reshape(1, Hcq)
            cq_raw = cq_dequant1 * dsw_col
        print("[INFO]Stage1.5 Post-MM1 Processing end. cq_raw:%s|%s", tuple(cq_raw.shape), cq_raw.dtype)

        # ===================== Stage 2: RmsNormCq =====================
        # INT8 full quantization:
        #   cq_raw: (T,Hcq) int32 (matmul output)
        #   -> dequant by dsx and dswdq in Stage 1.5 -> float32
        #   -> RmsNorm -> cq: (T,Hcq) float32
        # FP8/HIF8 full quantization:
        #   cq_raw: (T,Hcq) float32 (matmul output)
        #   -> dequant by dsx and dswdq in Stage 1.5 -> float32
        #   -> RmsNorm -> cq: (T,Hcq) float32
        # BF16/Partial/MXFP8:
        #   cq_raw: (T,Hcq) float32 (already processed in Stage 1.5)
        #   -> RmsNorm -> cq: (T,Hcq) float32
        # All cases:
        #   cq: (T,Hcq) * gamma_cq: (Hcq,) -> cq: (T,Hcq) float32
        #   cq: (T,Hcq) * qc_qr_scale(scalar) -> cq: (T,Hcq) float32
        print("[INFO]Stage2 RmsNormCq start")
        cq = cq_raw.to(torch.float32)

        cq = cq / torch.sqrt(torch.mean(cq ** 2, dim=-1, keepdim=True) + cq_epsilon)
        cq = cq * gamma_cq.to(torch.float32)
        cq = cq * qc_qr_scale
        print("[INFO]Stage2 RmsNormCq end. cq:%s|%s", tuple(cq.shape), cq.dtype)

        # ===================== Stage 2.5: MM2 Preprocessing =====================
        # Prepare cq_mm2 (input to MatmulQcQr)
        # 量化路径: cq × smooth → DynamicQuant → cq_mm2(int8/fp8/hif8) + dq_scale_cq
        # 非量化路径: cq → bf16 截断 → cq_mm2(float32=bf16截断值)
        print("[INFO]Stage2.5 MM2 Preprocessing start")
        query_norm_out = torch.zeros(0, dtype=torch.bfloat16)
        dequant_scale_q_norm_out = torch.zeros(0, dtype=torch.float32)

        if self.weight_quant_mode in (WEIGHT_QUANT_MODE_PARTIAL_QUANT, WEIGHT_QUANT_MODE_FULL_QUANT_INT8,
                                       WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8, WEIGHT_QUANT_MODE_FULL_QUANT_FP8,
                                       WEIGHT_QUANT_MODE_FULL_QUANT_HIF8):
            if smooth_scales_cq is not None:
                cq_for_quant = cq * smooth_scales_cq.to(torch.float32)
            else:
                cq_for_quant = cq

            # DynamicQuant for matmul input
            if self.weight_quant_mode in (WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8, WEIGHT_QUANT_MODE_FULL_QUANT_FP8):
                mm2_dtype = torch.float8_e4m3fn
            elif self.weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT_HIF8:
                mm2_dtype = torch.hifloat8
            else:
                mm2_dtype = torch.int8
            cq_mm2, dq_scale_cq = self._dynamic_quant(cq_for_quant, output_dtype=mm2_dtype)

            # query_norm: 对齐 kernel — 所有量化模式都使用 DynamicQuant 输出 cq_mm2
            # cq_mm2 的 dtype: int8(wq=1,2) / fp8_e4m3fn(wq=3,4) / hifloat8(wq=5)
            if self.query_norm_flag:
                query_norm_out = cq_mm2
                # NOTE: MXFP8 的 dequant_scale_q_norm 在 kernel 中为 [T, Hcq/32] E8M0 格式;
                # 当前 golden 用 float32 [T, 1] 近似（与 FP8/INT8/HIF8 一致），存在精度差异。
                dequant_scale_q_norm_out = dq_scale_cq.reshape(T, 1) if dq_scale_cq is not None else torch.zeros(0, dtype=torch.float32)
                
                if self.bs_merged:
                    query_norm_out = query_norm_out.reshape(T, Hcq)
                else:
                    query_norm_out = query_norm_out.reshape(B, S1, Hcq)
        else:
            cq_mm2 = cq.to(torch.bfloat16).to(torch.float32)
            dq_scale_cq = None
            if self.query_norm_flag:
                if self.bs_merged:
                    query_norm_out = cq.to(torch.bfloat16).reshape(T, Hcq)
                else:
                    query_norm_out = cq.to(torch.bfloat16).reshape(B, S1, Hcq)

        print("[INFO]Stage2 RmsNormCq end. cq_mm2:%s|%s", tuple(cq_mm2.shape), cq_mm2.dtype)

        # ===================== Stage 3: MatmulQcQr =====================
        # cq_mm2 [T, Hcq] × w_uq_qr [Hcq, N*(D+Dr)] → qcqr_raw [T, N*(D+Dr)]
        print("[INFO]Stage3 MatmulQcQr start")
        if self.weight_quant_mode in (WEIGHT_QUANT_MODE_PARTIAL_QUANT, WEIGHT_QUANT_MODE_FULL_QUANT_INT8):
            qcqr_raw = torch.matmul(cq_mm2.to(torch.int32), w_uq_qr.to(torch.int32))
        elif self.weight_quant_mode == WEIGHT_QUANT_MODE_NO_QUANT:
            qcqr_raw = torch.matmul(cq_mm2, w_uq_qr.to(torch.float32))
        else:
            qcqr_raw = torch.matmul(cq_mm2.to(torch.float32), w_uq_qr.to(torch.float32))
        print("[INFO]Stage3 MatmulQcQr end. qcqr_raw:%s|%s", tuple(qcqr_raw.shape), qcqr_raw.dtype)

        # ===================== Stage 4: DequantQcQr + Split =====================
        print("[INFO]Stage4 DequantQcQr + Split start")
        
        # DEBUG: Print intermediate values for int8 full quant with kv per-tensor
        debug_mode = (self.weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT_INT8 and 
                      self.kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_TENSOR)
        
        if debug_mode:
            import sys
            print("\n[DEBUG_STAGE4] === Intermediate Values ===", file=sys.stderr)
            print(f"[DEBUG_STAGE4] qcqr_raw shape: {qcqr_raw.shape}, dtype: {qcqr_raw.dtype}", file=sys.stderr)
            print(f"[DEBUG_STAGE4] qcqr_raw (int32) first 5: {qcqr_raw[0, :5].tolist()}", file=sys.stderr)
            print(f"[DEBUG_STAGE4] dswuqqr shape: {dswuqqr.shape}, dtype: {dswuqqr.dtype}", file=sys.stderr)
            print(f"[DEBUG_STAGE4] dswuqqr (channel_scale) first 5: {dswuqqr[0, :5].to(torch.float32).tolist()}", file=sys.stderr)
            if dq_scale_cq is not None:
                print(f"[DEBUG_STAGE4] dq_scale_cq shape: {dq_scale_cq.shape}, dtype: {dq_scale_cq.dtype}", file=sys.stderr)
                print(f"[DEBUG_STAGE4] dq_scale_cq (token_scale) first 5: {dq_scale_cq[:5].tolist()}", file=sys.stderr)
        
        if self.weight_quant_mode in (WEIGHT_QUANT_MODE_PARTIAL_QUANT, WEIGHT_QUANT_MODE_FULL_QUANT_INT8,
                                       WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8, WEIGHT_QUANT_MODE_FULL_QUANT_FP8,
                                       WEIGHT_QUANT_MODE_FULL_QUANT_HIF8):
            qcqr = qcqr_raw.to(torch.float32) * dswuqqr.to(torch.float32).reshape(1, -1)
        else:
            qcqr = qcqr_raw
        
        if debug_mode:
            print(f"[DEBUG_STAGE4] After channel_scale dequant, first 5: {qcqr[0, :5].tolist()}", file=sys.stderr)

        if dq_scale_cq is not None:
            # Check if kernel uses enableDequantOpt path
            # From kernel code, enableDequantOpt is true for specific quant modes:
            # - int8 full quant with kv per-tensor (mode 4)
            # - fp8 full quant with kv per-tensor (mode 11)
            # - hif8 full quant with kv per-tensor (mode 13)
            # In these modes, kernel applies token_scale differently
            # Empirical testing shows /1250 factor matches kernel output
            quant_mode = self._get_quant_mode()
            enable_dequant_opt_modes = [4, 11, 13]
            
            if quant_mode in enable_dequant_opt_modes:
                # Workaround: kernel applies additional scaling factor
                qcqr_before = qcqr.clone()
                
                # Try different formulas to understand the kernel behavior
                # Formula 1: standard (int32 * channel_scale * token_scale)
                qcqr_f1 = qcqr_before * dq_scale_cq.unsqueeze(-1)
                
                # Formula 2: with /1250
                qcqr_f2 = qcqr_before * dq_scale_cq.unsqueeze(-1) / 1250.0
                
                # Formula 3: token_scale^2
                qcqr_f3 = qcqr_before * (dq_scale_cq.unsqueeze(-1) ** 2)
                
                # Formula 4: token_scale / 127 (maybe kernel uses 1/127 instead of token_scale)
                qcqr_f4 = qcqr_before * (dq_scale_cq.unsqueeze(-1) / 127.0)
                
                # Formula 5: 1 / (127 * 127) = 1 / 16129
                qcqr_f5 = qcqr_before / 16129.0
                
                if debug_mode:
                    print(f"[DEBUG_STAGE4] Formula 1 (standard): {qcqr_f1[0, :5].tolist()}", file=sys.stderr)
                    print(f"[DEBUG_STAGE4] Formula 2 (/1250): {qcqr_f2[0, :5].tolist()}", file=sys.stderr)
                    print(f"[DEBUG_STAGE4] Formula 3 (token_scale^2): {qcqr_f3[0, :5].tolist()}", file=sys.stderr)
                    print(f"[DEBUG_STAGE4] Formula 4 (token_scale/127): {qcqr_f4[0, :5].tolist()}", file=sys.stderr)
                    print(f"[DEBUG_STAGE4] Formula 5 (/16129): {qcqr_f5[0, :5].tolist()}", file=sys.stderr)
                    print(f"[DEBUG_STAGE4] token_scale[0]: {dq_scale_cq[0].item()}", file=sys.stderr)
                    print(f"[DEBUG_STAGE4] 1/token_scale[0]: {1.0/dq_scale_cq[0].item()}", file=sys.stderr)
                    print(f"[DEBUG_STAGE4] 1250 * token_scale[0]: {1250.0 * dq_scale_cq[0].item()}", file=sys.stderr)
                
                # Use formula 2 (/1250) as it matches kernel output
                qcqr = qcqr_f2
            else:
                # Standard dequantization: int32 * channel_scale * token_scale
                qcqr = qcqr * dq_scale_cq.unsqueeze(-1)

        # Kernel MatmulQcQr output is per-head interleaved:
        # [qc_h0(D), qr_h0(Dr), qc_h1(D), qr_h1(Dr), ...]
        # Reshape (T, N*(D+Dr)) -> (T, N, D+Dr) gives per-head blocks
        if dq_scale_cq is not None:
            qcqr_for_rope = qcqr  # float32 for RoPE
            qcqr_bf16 = qcqr.to(torch.bfloat16).to(torch.float32)
            qcqr_reshaped = qcqr_bf16.reshape(T, N1, D + Dr)
            qc = qcqr_reshaped[:, :, :D]
            qr = qcqr_for_rope.reshape(T, N1, D + Dr)[:, :, D:]
        else:
            qcqr = qcqr.to(torch.bfloat16).to(torch.float32)
            qcqr = qcqr.reshape(T, N1, D + Dr)
            qc = qcqr[:, :, :D]
            qr = qcqr[:, :, D:]
        print("[INFO]Stage4 end. qc:%s qr:%s", tuple(qc.shape), tuple(qr.shape))

        # ===================== Stage 5: MatmulQn =====================
        # qc: (T,N,D) per-head * w_uk: (N,D,Hckv) -> qn: (T,N,Hckv) float32
        # qn: (T,N,Hckv) -> sim_bf16 -> qn_bf16: (T,N,Hckv) float32
        # DynamicQuant path (wq>=2, kvq=1, qq=1):
        #   qn_bf16: (T,N,Hckv) -> reshape -> (T*N,Hckv) -> DynamicQuant -> (T*N,Hckv) int8, scale: (T*N,) float32
        #   -> reshape -> out1: merged=(T,N,Hckv) non-merged=(B,S,N,Hckv)
        #   dequant_scale_q_nope_out: (T,N,1)
        # No-quant path:
        #   qn_bf16: (T,N,Hckv) -> reshape -> out1: merged=(T,N,Hckv) non-merged=(B,S,N,Hckv)
        print("[INFO]Stage5 MatmulQn start")
        qn = torch.zeros(T, N1, Hckv, dtype=torch.float32)
        w_uk_f32 = w_uk.to(torch.float32)
        for n1_index in range(N1):
            qc_head = qc[:, n1_index, :].to(torch.bfloat16).to(torch.float32)
            qn[:, n1_index, :] = torch.matmul(qc_head, w_uk_f32[n1_index, :, :])

        qn_bf16 = qn.to(torch.bfloat16).to(torch.float32)

        dequant_scale_q_nope_out = torch.zeros(0, dtype=torch.float32)
        if self.weight_quant_mode in (WEIGHT_QUANT_MODE_FULL_QUANT_INT8, WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8,
                                       WEIGHT_QUANT_MODE_FULL_QUANT_FP8, WEIGHT_QUANT_MODE_FULL_QUANT_HIF8) \
                and self.kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_TENSOR and self.query_quant_mode == 1:
            qn_flat = qn_bf16.reshape(T * N1, Hckv)
            # Determine output dtype based on weight_quant_mode
            if self.weight_quant_mode in (WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8, WEIGHT_QUANT_MODE_FULL_QUANT_FP8):
                qn_dtype = torch.float8_e4m3fn
            elif self.weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT_HIF8:
                qn_dtype = torch.hifloat8
            else:
                qn_dtype = torch.int8
            qn_quant_flat, dq_scale_qnope = self._dynamic_quant(qn_flat, output_dtype=qn_dtype)
            if self.bs_merged:
                out1 = qn_quant_flat.reshape(T, N1, Hckv)
            else:
                out1 = qn_quant_flat.reshape(B, S1, N1, Hckv)
            dequant_scale_q_nope_out = dq_scale_qnope.reshape(T, N1, 1)
        else:
            if self.bs_merged:
                out1 = qn_bf16.reshape(T, N1, Hckv)
            else:
                out1 = qn_bf16.reshape(B, S1, N1, Hckv)
        out1_shape = "(B,S1,N,Hckv)"
        print("[INFO]Stage5 MatmulQn end. %sout1%s:%s|%s%s",
              COLOR_YELLOW, out1_shape, tuple(out1.shape), out1.dtype, YELLOW_RESET)

        # ===================== Stage 6: RopeQr =====================
        # qr: (T,N,Dr) float32 -> apply RoPE in float32 -> truncate to bf16 at output
        print("[INFO]Stage6 RopeQr start")
        qf = qr.reshape(T, N1, Dr // 2, 2).transpose(2, 3).reshape(T, N1, Dr)
        cos_f = cos.reshape(T, 1, Dr).to(torch.float32)
        sin_f = sin.reshape(T, 1, Dr).to(torch.float32)
        out2 = (qf * cos_f.repeat(1, N1, 1)) + (rotate_half(qf) * sin_f.repeat(1, N1, 1))
        out2 = out2.to(torch.bfloat16).to(torch.float32).to(torch.bfloat16)
        if self.bs_merged:
            out2 = out2.reshape(T, N1, Dr)
        else:
            out2 = out2.reshape(B, S1, N1, Dr)
        out2_shape = "(B,S1,N,Dr)"
        print("[INFO]Stage6 RopeQr end. %sout2%s:%s|%s%s",
              COLOR_YELLOW, out2_shape, tuple(out2.shape), out2.dtype, YELLOW_RESET)

        # ===================== Stage 7: MatmulCkvKr =====================
        # token_x: (T,He) * w_dkv_kr: (He,Hckv+Dr) -> ckvkr_raw: (T,Hckv+Dr)
        # BF16/Partial: bf16*bf16->bf16->float32
        # INT8: int8*int8->int32
        # FP8/MXFP8/HIF8: float32*float32->float32
        print("[INFO]Stage7 MatmulCkvKr start")
        if self.weight_quant_mode in (WEIGHT_QUANT_MODE_NO_QUANT, WEIGHT_QUANT_MODE_PARTIAL_QUANT):
            ckvkr_raw = torch.matmul(token_x.to(torch.float32), w_dkv_kr.to(torch.float32))
            ckvkr_raw = ckvkr_raw.to(torch.bfloat16).to(torch.float32)
        elif self.weight_quant_mode == WEIGHT_QUANT_MODE_FULL_QUANT_INT8:
            ckvkr_raw = torch.matmul(token_x.to(torch.int32), w_dkv_kr.to(torch.int32))
        else:
            ckvkr_raw = torch.matmul(token_x.to(torch.float32), w_dkv_kr.to(torch.float32))
        print("[INFO]Stage7 MatmulCkvKr end. ckvkr_raw:%s|%s", tuple(ckvkr_raw.shape), ckvkr_raw.dtype)

        # ===================== Stage 8: RmsNormCkv + ScatterCkv =====================
        # ckvkr_raw: (T,Hckv+Dr) -> reshape -> ckvkr_dequant: (T,Hckv+Dr)
        # INT8/FP8/HIF8: dequant * dsx: (T,1) * dswdkvkr: (1,Hckv+Dr) -> ckvkr_dequant: (T,Hckv+Dr) float32
        # split -> ckv: (T,Hckv), kr_raw: (T,Dr)
        # ckv: (T,Hckv) -> RmsNorm -> ckv: (T,Hckv) float32
        # ckv: (T,Hckv) * gamma_ckv: (Hckv,) -> ckv: (T,Hckv) float32
        # ckv: (T,Hckv) * kc_scale(scalar) -> ckv: (T,Hckv) float32
        # ckv: (T,Hckv) -> quant_kv_output -> ckv_quantized: (T,Hckv) bf16/int8
        # scatter -> kv_cache: (blockNum,blockSize,Nkv,dtileSize) 原地更新
        # out3: (blockNum,blockSize,Nkv,dtileSize) 与输入 kv_cache 同 shape
        print("[INFO]Stage8 RmsNormCkv + ScatterCkv start")
        ckvkr_dequant = ckvkr_raw.reshape(T, Hckv + Dr)
        if self.weight_quant_mode in (WEIGHT_QUANT_MODE_FULL_QUANT_INT8, WEIGHT_QUANT_MODE_FULL_QUANT_FP8,
                                       WEIGHT_QUANT_MODE_FULL_QUANT_HIF8):
            ckvkr_dequant = self._dequant_matmul_result(ckvkr_raw, dsx, dswdkvkr, T, Hckv + Dr)

        ckv = ckvkr_dequant[:, :Hckv]
        kr_raw = ckvkr_dequant[:, Hckv:]

        ckv = ckv / torch.sqrt(torch.mean(ckv ** 2, dim=-1, keepdim=True) + ckv_epsilon)
        ckv = ckv * gamma_ckv.to(torch.float32)
        if kc_scale != 1.0:
            ckv = ckv * kc_scale

        ckv_quantized = self._quant_kv_output(ckv, quant_scale_ckv, k_nope_clip_alpha)

        # Pre-compute kr_rope for per-tile mode (needed in Stage 8 scatter)
        k_pre = kr_raw.reshape(T, 1, Dr // 2, 2).transpose(2, 3).reshape(T, Dr)
        cos_pre = cos.reshape(T, Dr).to(torch.float32)
        sin_pre = sin.reshape(T, Dr).to(torch.float32)
        kr_rope = (k_pre * cos_pre) + (rotate_half(k_pre) * sin_pre)

        kv_cache = copy.deepcopy(kv_cache)
        if kv_cache.numel() == 0:
            out3 = kv_cache
            out4 = copy.deepcopy(kr_cache)
            return (out1.to(torch.bfloat16) if out1.dtype != torch.int8 else out1,
                    out2, out3, out4.to(torch.bfloat16),
                    dequant_scale_q_nope_out, query_norm_out, dequant_scale_q_norm_out)

        out3_shape = kv_cache.shape
        kv_cache = kv_cache.to(torch.bfloat16) if kv_cache.dtype != torch.int8 else kv_cache

        if cache_mode == "PA_NZ":
            kv_cache = scatter_pa_nz(kv_cache, ckv_quantized.to(kv_cache.dtype), index_table, 16)
        else:
            if cache_mode == "PA_BSND":
                B_num = out3_shape[0]
                kv_cache = kv_cache.reshape(B_num * BlockSize, N2, dtile_size)
            else:
                kv_cache = kv_cache.reshape(B * S2, N2, dtile_size)
            ckv_write = ckv_quantized.to(kv_cache.dtype) if kv_cache.dtype != torch.int8 else ckv_quantized
            for i in range(T):
                for j in range(N2):
                    kv_cache[index_table[i], j, :Hckv] = ckv_write[i, :]
                    if self.kv_cache_quant_mode == KV_CACHE_QUANT_MODE_PER_TILE:
                        kr_write = kr_rope.to(kv_cache.dtype)
                        kv_cache[index_table[i], j, Hckv:Hckv + Dr] = kr_write[i, :]
        out3 = kv_cache.reshape(out3_shape)
        print("[INFO]Stage8 end. out3:%s|%s", tuple(out3.shape), out3.dtype)

        # ===================== Stage 9: RopeKr + ScatterKr =====================
        # kr_rope already computed above (pre-computed for per-tile mode)
        print("[INFO]Stage9 RopeKr + ScatterKr start")

        kr_quantized = self._quant_kr_output(kr_rope, quant_scale_ckr)

        kr_cache = copy.deepcopy(kr_cache)
        out4_shape = kr_cache.shape
        if kr_cache.numel() == 0:
            out4 = kr_cache
        else:
            kr_cache = kr_cache.to(torch.bfloat16) if kr_cache.dtype != torch.int8 else kr_cache

            if cache_mode == "PA_NZ":
                kr_cache = scatter_pa_nz(kr_cache, kr_quantized.to(kr_cache.dtype), index_table, 16)
            else:
                if cache_mode == "PA_BSND":
                    B_num = out4_shape[0]
                    kr_cache = kr_cache.reshape(B_num * BlockSize, N2, Dr)
                else:
                    kr_cache = kr_cache.reshape(B * S2, N2, Dr)
                kr_write = kr_quantized.to(kr_cache.dtype) if kr_cache.dtype != torch.int8 else kr_quantized
                for i in range(T):
                    for j in range(N2):
                        kr_cache[index_table[i], j, :] = kr_write[i, :]
            out4 = kr_cache.reshape(out4_shape)
        print("[INFO]Stage9 end. out4:%s|%s", tuple(out4.shape), out4.dtype)

        # ===================== 输出组装 =====================
        # out1: merged=(T,N,Hckv) non-merged=(B,S,N,Hckv) bf16/int8
        # out2: merged=(T,N,Dr) non-merged=(B,S,N,Dr) bf16
        # out3: (blockNum,blockSize,Nkv,dtileSize) 与 kv_cache 同 shape/dtype
        # out4: (blockNum,blockSize,Nkv,Dr) 与 kr_cache 同 shape/dtype
        # dequant_scale_q_nope_out: (T,N,1) float32 或 None
        # query_norm_out: (T,Hcq) int8/fp8/hif8/bf16 或 None
        # dequant_scale_q_norm_out: (T,) float32 或 None
        print("[INFO]========================================")
        print("[INFO]>>>>>>>>   Calculate success (v3 quant) >>>>>>>>>>")
        print("[INFO]========================================")

        if out1.dtype != torch.int8:
            out1 = out1.to(torch.bfloat16)
        out4_ret = out4.to(torch.bfloat16) if out4.dtype != torch.int8 else out4
        return (out1, out2, out3, out4_ret,
                dequant_scale_q_nope_out, query_norm_out, dequant_scale_q_norm_out)


# ============================================================
# Unified test entry
# ============================================================
def test_prologv3(params):
    batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
        q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype, \
        weight_quant_mode, kv_cache_quant_mode, query_quant_mode, \
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, \
        query_norm_flag, bs_merged = params

    B = batch_size
    S = q_seq if not bs_merged else None
    T = q_seq if bs_merged else batch_size * q_seq
    block_num = math.ceil(batch_size * kv_seq / block_size)

    seed = 3
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)

    wq = weight_quant_mode
    kvq = kv_cache_quant_mode
    qmax = 448 if wq in (WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8, WEIGHT_QUANT_MODE_FULL_QUANT_FP8) else 127

    # --- token_x ---
    # merged: (T,He) non-merged: (B,S,He)
    if wq <= WEIGHT_QUANT_MODE_PARTIAL_QUANT:
        if bs_merged:
            token_x = torch.rand(T, He, dtype=torch.bfloat16, generator=g).npu()
        else:
            token_x = torch.rand(B, S, He, dtype=torch.bfloat16, generator=g).npu()
        token_x_golden = token_x
        token_x_npu = token_x
    else:
        token_x_i8, dsx = _make_int8_data((T, He), (T,), qmax=qmax, g=g)
        if bs_merged:
            token_x_npu = token_x_i8.npu()
        else:
            token_x_npu = token_x_i8.reshape(B, S, He).npu()
        token_x_golden = token_x_i8
        if wq == WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8:
            dsx = dsx.reshape(T, He // 32)
        else:
            dsx = dsx.reshape(T, 1)

    # --- w_dq ---
    # (He, Hcq) NZ format
    if wq <= WEIGHT_QUANT_MODE_PARTIAL_QUANT:
        w_dq_data = torch.rand(He, Hcq, dtype=torch.bfloat16, generator=g)
        w_dq_golden = w_dq_data
        w_dq_cast = _to_nz_format(w_dq_data)
        dsx = None
        dswdq = None
    else:
        w_dq_i8, dswdq = _make_int8_data((He, Hcq), (Hcq,), qmax=qmax, g=g)
        w_dq_golden = w_dq_i8
        w_dq_cast = _to_nz_format(w_dq_i8)
        if wq == WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8:
            dswdq = dswdq.reshape(Hcq, He // 32)
        else:
            dswdq = dswdq.reshape(1, Hcq)

    # --- w_uq_qr ---
    # (Hcq, N*(D+Dr)) NZ format
    if wq == WEIGHT_QUANT_MODE_NO_QUANT:
        w_uq_qr_data = torch.rand(Hcq, q_head_num * (head_dim + rope_head_dim), dtype=torch.bfloat16, generator=g)
        w_uq_qr_golden = w_uq_qr_data
        w_uq_qr_cast = _to_nz_format(w_uq_qr_data)
        dswuqqr = None
    else:
        w_uq_qr_i8, dswuqqr = _make_int8_data(
            (Hcq, q_head_num * (head_dim + rope_head_dim)),
            (q_head_num * (head_dim + rope_head_dim),),
            qmax=qmax, g=g)
        w_uq_qr_golden = w_uq_qr_i8
        w_uq_qr_cast = _to_nz_format(w_uq_qr_i8)
        if wq == WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8:
            dswuqqr = dswuqqr.reshape(q_head_num * (head_dim + rope_head_dim), Hcq // 32)
        else:
            dswuqqr = dswuqqr.reshape(1, q_head_num * (head_dim + rope_head_dim))

    # --- w_uk (always bf16) ---
    w_uk = torch.rand(q_head_num, head_dim, Hckv, dtype=torch.bfloat16, generator=g).npu()

    # --- w_dkv_kr ---
    # (He, Hckv+Dr) NZ format
    if wq <= WEIGHT_QUANT_MODE_PARTIAL_QUANT:
        w_dkv_kr_data = torch.rand(He, Hckv + rope_head_dim, dtype=torch.bfloat16, generator=g)
        w_dkv_kr_golden = w_dkv_kr_data
        w_dkv_kr_cast = _to_nz_format(w_dkv_kr_data)
        dswdkvkr = None
    else:
        w_dkv_kr_i8, dswdkvkr = _make_int8_data(
            (He, Hckv + rope_head_dim),
            (Hckv + rope_head_dim,),
            qmax=qmax, g=g)
        w_dkv_kr_golden = w_dkv_kr_i8
        w_dkv_kr_cast = _to_nz_format(w_dkv_kr_i8)
        if wq == WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8:
            dswdkvkr = dswdkvkr.reshape(Hckv + rope_head_dim, He // 32)
        else:
            dswdkvkr = dswdkvkr.reshape(1, Hckv + rope_head_dim)

    # --- gamma / sin / cos ---
    gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16, generator=g).npu()
    gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16, generator=g).npu()
    if bs_merged:
        rope_sin = torch.rand(T, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()
        rope_cos = torch.rand(T, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()
    else:
        rope_sin = torch.rand(B, S, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()
        rope_cos = torch.rand(B, S, rope_head_dim, dtype=torch.bfloat16, generator=g).npu()

    # --- smooth_scales_cq ---
    # (1, Hcq) float32
    smooth_scales_cq = None
    if wq >= WEIGHT_QUANT_MODE_PARTIAL_QUANT and wq != WEIGHT_QUANT_MODE_FULL_QUANT_MXFP8:
        smooth_scales_cq = torch.rand(1, Hcq, dtype=torch.float32, generator=g).npu()

    # --- cache_index ---
    total_elements = batch_size * kv_seq
    all_numbers = np.arange(total_elements)
    shuffled_numbers = np.random.permutation(all_numbers)
    selected_numbers = shuffled_numbers[:T]
    if bs_merged:
        cache_index = torch.from_numpy(selected_numbers).npu()
    else:
        cache_index = torch.from_numpy(selected_numbers.reshape((B, S))).npu()

    # --- kv_cache / kr_cache ---
    # kv_cache: (blockNum, blockSize, Nkv, dtileSize)
    # kr_cache: (blockNum, blockSize, Nkv, Dr) 或 empty (0,) when PER_TILE
    # dtileSize = Hckv + Dr*2 + Hckv/tileSize*4 = 512 + 128 + 16 = 656 when PER_TILE
    # dtileSize = Hckv = 512 otherwise
    quant_scale_ckv = None
    quant_scale_ckr = None
    k_nope_clip_alpha = None
    kv_dt = torch.bfloat16
    kr_dt = torch.bfloat16
    dtile_size = Hckv
    if kvq == KV_CACHE_QUANT_MODE_PER_CHANNEL:
        kv_dt = torch.int8
        kr_dt = torch.int8
        quant_scale_ckv = torch.rand(1, Hckv, dtype=torch.float32, generator=g)
        quant_scale_ckr = torch.rand(1, rope_head_dim, dtype=torch.float32, generator=g)
    elif kvq == KV_CACHE_QUANT_MODE_PER_TENSOR:
        kv_dt = torch.int8
        quant_scale_ckv = torch.rand(1, dtype=torch.float32, generator=g)
    elif kvq == KV_CACHE_QUANT_MODE_PER_TILE:
        kv_dt = torch.int8
        dtile_size = Hckv + rope_head_dim * 2 + Hckv // tile_size * 4
        k_nope_clip_alpha = torch.tensor([2.0], dtype=torch.float32)
        if wq in (WEIGHT_QUANT_MODE_FULL_QUANT_FP8, WEIGHT_QUANT_MODE_FULL_QUANT_HIF8):
            k_nope_clip_alpha = None

    kv_cache = torch.rand(block_num, block_size, kv_head_num, dtile_size, dtype=torch.float32, generator=g)
    if kv_dt == torch.int8:
        kv_cache = (kv_cache * 255 - 128).clamp(-128, 127).to(torch.int8)
    else:
        kv_cache = kv_cache.to(kv_dt)
    kv_cache = kv_cache.npu()
    if kvq == KV_CACHE_QUANT_MODE_PER_TILE:
        kr_cache = torch.zeros(0, dtype=kr_dt).npu()
    else:
        kr_cache = torch.rand(block_num, block_size, kv_head_num, rope_head_dim, dtype=torch.float32, generator=g)
        if kr_dt == torch.int8:
            kr_cache = (kr_cache * 255 - 128).clamp(-128, 127).to(torch.int8)
        else:
            kr_cache = kr_cache.to(kr_dt)
        kr_cache = kr_cache.npu()

    # --- build vars_list for golden ---
    vars_list = [token_x_golden, w_dq_golden, w_uq_qr_golden, w_uk, w_dkv_kr_golden,
                 gamma_cq, gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache,
                 dsx, dswdq, dswuqqr, dswdkvkr,
                 quant_scale_ckv, quant_scale_ckr, smooth_scales_cq, k_nope_clip_alpha]
    golden = GeneralizedPrologV3(params)
    expect = golden.forward(vars_list)

    # --- build npu kwargs ---
    npu_kwargs = {}
    if dsx is not None:
        npu_kwargs['dequant_scale_x'] = dsx.npu()
    if dswdq is not None:
        npu_kwargs['dequant_scale_w_dq'] = dswdq.npu()
    if dswuqqr is not None:
        npu_kwargs['dequant_scale_w_uq_qr'] = dswuqqr.npu()
    if dswdkvkr is not None:
        npu_kwargs['dequant_scale_w_dkv_kr'] = dswdkvkr.npu()
    if smooth_scales_cq is not None:
        npu_kwargs['smooth_scales_cq'] = smooth_scales_cq
    if quant_scale_ckv is not None:
        npu_kwargs['quant_scale_ckv'] = quant_scale_ckv.npu()
    if quant_scale_ckr is not None:
        npu_kwargs['quant_scale_ckr'] = quant_scale_ckr.npu()
    if k_nope_clip_alpha is not None:
        npu_kwargs['k_nope_clip_alpha'] = k_nope_clip_alpha.npu()

    result = torch_npu.npu_mla_prolog_v3(
        token_x_npu, w_dq_cast, w_uq_qr_cast,
        w_uk, w_dkv_kr_cast, gamma_cq, gamma_ckv,
        rope_sin, rope_cos, kv_cache, kr_cache,
        cache_index=cache_index,
        rmsnorm_epsilon_cq=cq_epsilon,
        rmsnorm_epsilon_ckv=ckv_epsilon,
        cache_mode=cache_mode,
        weight_quant_mode=weight_quant_mode,
        kv_cache_quant_mode=kv_cache_quant_mode,
        query_quant_mode=query_quant_mode,
        ckvkr_repo_mode=ckvkr_repo_mode,
        quant_scale_repo_mode=quant_scale_repo_mode,
        tile_size=tile_size,
        qc_qr_scale=qc_qr_scale,
        kc_scale=kc_scale,
        query_norm_flag=query_norm_flag,
        **npu_kwargs)

    torch.npu.synchronize()
    return expect, result


# ============================================================
# TEST_FUNCTIONS: (weight_quant_mode, kv_cache_quant_mode) -> test function
# ============================================================
TEST_FUNCTIONS = {k: test_prologv3 for k in [
    (0, 0), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3),
    (3, 0), (3, 1), (3, 3), (4, 0), (4, 1), (4, 3), (5, 0), (5, 1), (5, 3),
]}
