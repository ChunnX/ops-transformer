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
COLOR_GREEN = "\033[32m"
GREEN_RESET = "\033[0m"


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.concatenate((-x2, x1), dim=-1)


class GeneralizedPrologV3:
    def __init__(self, params):
        self.batch_size, self.He, self.Hcq, self.Hckv, self.q_head_num, self.kv_head_num, \
            self.head_dim, self.rope_head_dim, \
            self.q_seq, self.kv_seq, self.block_size, self.input_layout, \
            self.cache_mode, self.cq_epsilon, self.ckv_epsilon, self.dtype, \
            self.weight_quant_mode, self.kv_cache_quant_mode, self.query_quant_mode, \
            self.ckvkr_repo_mode, self.quant_scale_repo_mode, self.tile_size, \
            self.qc_qr_scale, self.kc_scale, self.query_norm_flag = params

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

        vars_list = [t.cpu() for t in vars_list]
        (token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv,
         sin, cos, index_table, kv_cache, kr_cache) = vars_list
        cq_epsilon = self.cq_epsilon
        ckv_epsilon = self.ckv_epsilon
        cache_mode = self.cache_mode

        print("[INFO]========================================")
        print("[INFO]>>>>>>>>  Start to calculate (v3) >>>>>>>>>>")
        print("[INFO]========================================")

        # matmul1 : token_x(B*S1,He) * w_dq(He,Hcq) -> matmul1_res(B*S1,Hcq)
        matmul1_dtype = torch.float32
        x_shape = "(B*S1,He)"
        print("[INFO]matmul1 start. token_x%s:%s|%s w_dq(He,Hcq):%s|%s matmul1_dtype:%s",
              x_shape, tuple(token_x.shape), token_x.dtype, tuple(w_dq.shape), w_dq.dtype, matmul1_dtype)
        matmul1_res = torch.matmul(token_x, w_dq).to(matmul1_dtype)
        print("[INFO]cast matmul1_res->bfloat16->float32")
        matmul1_res = matmul1_res.to(torch.bfloat16).to(torch.float32)
        matmul1_res_shape = "(B*S1,Hcq)"
        print("[INFO]matmul1 end. matmul1_res%s:%s|%s", matmul1_res_shape, tuple(matmul1_res.shape), matmul1_res.dtype)

        # rmsnorm1 : matmul1_res(B*S1,Hcq) * gamma_cq(Hcq) -> norm1_res(B*S1,Hcq)
        ep1 = float(cq_epsilon)
        gamma1 = gamma_cq
        print("[INFO]rmsnorm1 start. matmul1_res%s:%s|%s gamma_cq(Hcq):%s|%s",
              matmul1_res_shape, tuple(matmul1_res.shape), matmul1_res.dtype,
              tuple(gamma1.shape), gamma1.dtype)
        norm1_res = matmul1_res / torch.sqrt(torch.mean(matmul1_res ** 2, dim=-1, keepdim=True) + ep1)
        norm1_res *= gamma1
        print("[INFO]rmsnorm1 end. norm1_res%s:%s|%s",
              matmul1_res_shape, tuple(norm1_res.shape), norm1_res.dtype)

        # matmul2 : norm1_res(B*S1,Hcq) * w_uq_qr(Hcq,N*(D+Dr)) -> matmul2_res(B*S1,N,(D+Dr))
        matmul2_dtype = torch.float32
        print("[INFO]cast norm1_res->bfloat16->float32")
        norm1_res = norm1_res.to(torch.bfloat16).to(torch.float32)
        w_uq_qr = w_uq_qr.to(torch.float32)
        print("[INFO]matmul2 start. norm1_res%s:%s|%s w_uq_qr(Hcq,N*(D+Dr)):%s|%s matmul2_dtype:%s",
              matmul1_res_shape, tuple(norm1_res.shape), norm1_res.dtype,
              tuple(w_uq_qr.shape), w_uq_qr.dtype, matmul2_dtype)
        matmul2_res = torch.matmul(norm1_res, w_uq_qr).to(matmul2_dtype)
        print("[INFO]cast matmul2_res->bfloat16->float32")
        matmul2_res = matmul2_res.to(torch.bfloat16).to(torch.float32)
        matmul2_res = matmul2_res.reshape(T, N1, D + Dr)
        print("[INFO]matmul2 end. matmul2_res(B*S1,N,D+Dr):%s|%s", tuple(matmul2_res.shape), matmul2_res.dtype)

        # splitD1 : matmul2_res(B*S1,N,D+Dr) -> splitd1_res1(B*S1,N,D) & splitd1_res2(B*S1,N,Dr)
        splitd1_res1 = matmul2_res[:, :, :D]
        splitd1_res2 = matmul2_res[:, :, D:]
        splitd1_res1_shape = "(B*S1,N1,D)"
        splitd1_res2_shape = "(B*S1,N1,Dr)"
        print("[INFO]splitD1 end. splitd1_res1%s:%s splitd1_res2%s:%s",
              splitd1_res1_shape, tuple(splitd1_res1.shape),
              splitd1_res2_shape, tuple(splitd1_res2.shape))

        # matmul3 : splitd1_res1(B*S1,N,D) * w_uk(N,D,Hckv) -> out1(B,S1,N,Hckv)
        splitd1_res1 = splitd1_res1.transpose(0, 1)
        out1 = torch.zeros((N1, T, Hckv))
        matmul3_dtype = torch.float32
        print("[INFO]cast w_uk->float32")
        w_uk = w_uk.to(torch.float32)
        splitd1_res1 = splitd1_res1.to(torch.bfloat16).to(torch.float32)
        print("[INFO]matmul3 start. splitd1_res1(B,S1,N,D):%s|%s w_uk(N,D,Hckv):%s|%s matmul3_dtype:%s",
              tuple(splitd1_res1.shape), splitd1_res1.dtype,
              tuple(w_uk.shape), w_uk.dtype, matmul3_dtype)
        for n1_index in range(N1):
            out1[n1_index, :, :] = torch.matmul(splitd1_res1[n1_index, :, :], w_uk[n1_index, :, :]).to(matmul3_dtype)
        out1 = out1.transpose(0, 1)
        out1 = out1.reshape(B, S1, N1, Hckv)
        out1_shape = "(B,S1,N,Hckv)"
        print("[INFO]matmul3 end. %sout1%s:%s|%s%s",
              COLOR_YELLOW, out1_shape, out1.shape, out1.dtype, YELLOW_RESET)

        # rotary1 : splitd1_res2(B*S1,N,Dr) * cos * sin -> out2(B,S1,N,Dr)
        cos_shape = "(B*S1,Dr)"
        print("[INFO]rotary1 start. splitd1_res2%s:%s|%s cos%s:%s|%s",
              splitd1_res2_shape, tuple(splitd1_res2.shape), splitd1_res2.dtype,
              cos_shape, tuple(cos.shape), cos.dtype)

        expanded_cos = cos.repeat(1, N1, 1)
        expanded_sin = sin.repeat(1, N1, 1)
        q = splitd1_res2.reshape(T, N1, int(Dr / 2), 2).transpose(3, 2).reshape(T, N1, Dr)
        out2 = (q * expanded_cos) + (rotate_half(q) * expanded_sin)
        out2 = out2.reshape(B, S1, N1, Dr)
        out2_shape = "(B,S1,N1,Dr)"
        print("[INFO]rotary1 end. %sout2%s:%s|%s%s",
              COLOR_YELLOW, out2_shape, tuple(out2.shape), out2.dtype, YELLOW_RESET)

        # matmul4 : token_x(B*S1,He) * w_kv_kr(He,Hckv+Dr) -> matmul4_res(B*S1,Hckv+Dr)
        w_kv_kr = w_dkv_kr
        matmul4_dtype = torch.float32
        print("[INFO]matmul4 start. token_x%s:%s|%s w_kv_kr(He,Hckv+Dr):%s|%s matmul4_dtype:%s",
              x_shape, tuple(token_x.shape), token_x.dtype,
              tuple(w_kv_kr.shape), w_kv_kr.dtype, matmul4_dtype)
        matmul4_res = torch.matmul(token_x, w_kv_kr).to(matmul4_dtype)
        print("[INFO]cast matmul4_res->bfloat16->float32")
        matmul4_res = matmul4_res.to(torch.bfloat16).to(torch.float32)
        matmul4_res_shape = "(B*S1,Hckv+Dr)"
        print("[INFO]matmul4 end. matmul4_res%s:%s|%s",
              matmul4_res_shape, tuple(matmul4_res.shape), matmul4_res.dtype)

        # splitD2 : matmul4_res(B*S1,Hckv+Dr) -> splitd2_res1(B*S1,Hckv) & splitd2_res2(B*S1,Dr)
        splitd2_res1 = matmul4_res.reshape(B * S1, Hckv + Dr)[:, :Hckv]
        splitd2_res2 = matmul4_res.reshape(B * S1, Hckv + Dr)[:, Hckv:]
        splitd2_res1_shape = "(B*S1,Hckv)"
        splitd2_res2_shape = "(B*S1,Dr)"
        print("[INFO]splitD2 end. splitd2_res1%s:%s, splitd2_res2%s:%s",
              splitd2_res1_shape, tuple(splitd2_res1.shape),
              splitd2_res2_shape, tuple(splitd2_res2.shape))

        # rmsnorm2 : splitd2_res1(B*S1,Hckv) * gamma_ckv(Hckv) -> norm2_res(B*S1,Hckv)
        ep2 = float(ckv_epsilon)
        gamma2 = gamma_ckv
        print("[INFO]rmsnorm2 start. splitd2_res1%s:%s|%s gamma_ckv(Hckv):%s|%s",
              splitd2_res1_shape, tuple(splitd2_res1.shape), splitd2_res1.dtype,
              tuple(gamma2.shape), gamma2.dtype)
        norm2_res = splitd2_res1 / torch.sqrt(torch.mean(splitd2_res1 ** 2, dim=-1, keepdim=True) + ep2)
        norm2_res *= gamma2
        print("[INFO]rmsnorm2 end. norm2_res%s:%s|%s",
              splitd2_res1_shape, tuple(norm2_res.shape), norm2_res.dtype)

        # scatter1 : norm2_res -> out3
        kv_cache = copy.deepcopy(kv_cache)

        if kv_cache.numel() == 0:
            out3 = kv_cache
            out4 = copy.deepcopy(kr_cache)
            return out1, out2, out3, out4

        out3_shape = kv_cache.shape
        out3_info = "(B,B,N2,Hckv)"
        scatter_size = 16
        kv_cache = kv_cache.to(torch.bfloat16)
        print("[INFO]scatter1 start. norm2_res%s:%s|%s kv_cache%s:%s|%s scatter_size:%d",
              splitd2_res1_shape, tuple(norm2_res.shape), norm2_res.dtype,
              out3_info, tuple(kv_cache.shape), kv_cache.dtype, scatter_size)
        if cache_mode == "PA_NZ":
            kv_cache = scatter_pa_nz(kv_cache, norm2_res, index_table, scatter_size)
        else:
            if cache_mode == "PA_BSND":
                B_num = out3_shape[0]
                kv_cache = kv_cache.reshape(B_num * BlockSize, N2, Hckv)
            else:
                kv_cache = kv_cache.reshape(B * S2, N2, Hckv)
            norm2_res = norm2_res.to(torch.bfloat16)
            for i in range(T):
                for j in range(N2):
                    kv_cache[index_table[i], j, :] = norm2_res[i, :]
        out3 = kv_cache.reshape(out3_shape)
        print("[INFO]scatter1 end. %sout3%s:%s|%s%s",
              COLOR_YELLOW, out3_info, tuple(out3.shape), out3.dtype, YELLOW_RESET)

        # rotary2 : splitd2_res2 * cos * sin -> rotary2_res
        print("[INFO]rotary2 start. splitd2_res2%s:%s|%s cos%s:%s|%s",
              splitd2_res2_shape, tuple(splitd2_res2.shape), splitd2_res2.dtype,
              cos_shape, tuple(cos.shape), cos.dtype)
        cos = cos.reshape(B * S1, Dr)
        sin = sin.reshape(B * S1, Dr)
        k = splitd2_res2.reshape(T, 1, int(Dr / 2), 2).transpose(3, 2).reshape(T, Dr)
        rotary2_res = (k * cos) + (rotate_half(k) * sin)
        print("[INFO]rotary2 end. rotary2_res%s:%s|%s", splitd2_res2_shape, tuple(rotary2_res.shape), rotary2_res.dtype)

        # scatter2 : rotary2_res -> out4
        kr_cache = copy.deepcopy(kr_cache)
        out4_shape = kr_cache.shape
        out4_info = "(B,B,N2,Dr)"
        scatter_size = 16
        kr_cache = kr_cache.to(torch.bfloat16)
        print("[INFO]scatter2 start. rotary2_res%s:%s|%s kr_cache%s:%s|%s scatter_size:%d",
              splitd2_res2_shape, tuple(rotary2_res.shape), rotary2_res.dtype,
              out4_info, tuple(kr_cache.shape), kr_cache.dtype, scatter_size)
        if cache_mode == "PA_NZ":
            kr_cache = scatter_pa_nz(kr_cache, rotary2_res, index_table, scatter_size)
        else:
            if cache_mode == "PA_BSND":
                B_num = out4_shape[0]
                kr_cache = kr_cache.reshape(B_num * BlockSize, N2, Dr)
            else:
                kr_cache = kr_cache.reshape(B * S2, N2, Dr)
            rotary2_res = rotary2_res.to(torch.bfloat16)
            for i in range(T):
                for j in range(N2):
                    kr_cache[index_table[i], j, :] = rotary2_res[i, :]
        out4 = kr_cache.reshape(out4_shape)
        print("[INFO]scatter2 end. %sout4%s:%s|%s%s",
              COLOR_YELLOW, out4_info, tuple(out4.shape), out4.dtype, YELLOW_RESET)

        print("[INFO]========================================")
        print("[INFO]>>>>>>>>   Calculate success (v3) >>>>>>>>>>")
        print("[INFO]========================================")

        out1 = out1.to(torch.bfloat16)
        out2 = out2.to(torch.bfloat16)
        return out1, out2, out3, out4


def scatter_pa_nz(kv_cache, norm2_res, index_table, scatter_size):
    B_num = kv_cache.shape[0]
    BlockSize = kv_cache.shape[1]
    kv_cache = kv_cache.reshape(B_num * BlockSize, kv_cache.shape[2], kv_cache.shape[3])

    T = index_table.shape[0]
    for i in range(0, T, scatter_size):
        cur_scatter_size = min(scatter_size, T - i)
        src = norm2_res[i:i + cur_scatter_size]
        indices = index_table[i:i + cur_scatter_size]
        kv_cache[indices] = src

    kv_cache = kv_cache.reshape(B_num, BlockSize, kv_cache.shape[1], kv_cache.shape[2])
    return kv_cache


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_prologv3_no_quant(params):
    batch_size, He, Hcq, Hckv, q_head_num, kv_head_num, head_dim, rope_head_dim, \
        q_seq, kv_seq, block_size, input_layout, cache_mode, cq_epsilon, ckv_epsilon, dtype, \
        weight_quant_mode, kv_cache_quant_mode, query_quant_mode, \
        ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale, query_norm_flag = params

    block_num = math.ceil(batch_size * kv_seq / block_size)

    seed = 3
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    token_x = torch.rand(batch_size, q_seq, He, dtype=torch.bfloat16, generator=generator).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16, generator=generator).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, q_head_num * (head_dim + rope_head_dim), dtype=torch.bfloat16, generator=generator).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(q_head_num, head_dim, Hckv, dtype=torch.bfloat16, generator=generator).npu()
    w_dkv_kr = torch.rand(He, Hckv + rope_head_dim, dtype=torch.bfloat16, generator=generator).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16, generator=generator).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16, generator=generator).npu()
    rope_sin = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=generator).npu()
    rope_cos = torch.rand(batch_size, q_seq, rope_head_dim, dtype=torch.bfloat16, generator=generator).npu()

    total_elements = batch_size * kv_seq
    all_numbers = np.arange(total_elements)
    shuffled_numbers = np.random.permutation(all_numbers)
    selected_numbers = shuffled_numbers[:batch_size * q_seq]
    cache_index_np = selected_numbers.reshape((batch_size, q_seq))
    cache_index = torch.from_numpy(cache_index_np).npu()

    kv_cache = torch.rand(block_num, block_size, kv_head_num, Hckv, dtype=torch.bfloat16, generator=generator).npu()
    kr_cache = torch.rand(block_num, block_size, kv_head_num, rope_head_dim, dtype=torch.bfloat16, generator=generator).npu()

    vars_list = [token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
                 rope_sin, rope_cos, cache_index, kv_cache, kr_cache]
    test_prologv3 = GeneralizedPrologV3(params)
    expect = test_prologv3.forward(vars_list)
    print("expect shapes:", expect[0].shape, expect[1].shape, expect[2].shape, expect[3].shape)

    result = torch_npu.npu_mla_prolog_v3(
        token_x, w_dq_cast, w_uq_qr_cast,
        w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq, rmsnorm_gamma_ckv,
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
        query_norm_flag=query_norm_flag)

    torch.npu.synchronize()
    print("result shapes:", result[0].shape, result[1].shape, result[2].shape, result[3].shape)
    return expect, result
