/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * \file twin_reference.cpp
 * \brief CPU 孪生参考实现的 C-ABI 出口（本地 ctypes 单测 / gtest 共用）。
 * 量化语义经 extern "C" 引用 quant 层唯一定义，本文件不复制任何量化逻辑。
 */

#include "twin_reference.h"

// quant 层 C-ABI 声明已由 quant_common.h 提供（唯一点）

extern "C" {

/**
 * CPU 孪生全流水入口。参数口径与 TwinIo 一一对应；返回 0 成功。
 * lse 传 nullptr 表示 return_lse=False；key_mean 传 nullptr 且 smooth_k=1 时内部自算 km。
 */
int SageAttnTwinRunForward(
    const void* q, const void* k, const void* v,
    const float* key_mean, void* out, float* lse,
    unsigned b, unsigned hq, unsigned hkv, unsigned sq, unsigned sk, unsigned d_og,
    unsigned tensor_layout, unsigned qk_quant_gran, unsigned pv_accum_dtype,
    unsigned is_causal, unsigned smooth_k, int dtype_mode, float sm_scale) {
    sage_attn::twin::TwinIo io{};
    io.q = q;
    io.k = k;
    io.v = v;
    io.key_mean = key_mean;
    io.out = out;
    io.lse = lse;
    io.b = b;
    io.hq = hq;
    io.hkv = hkv;
    io.sq = sq;
    io.sk = sk;
    io.d_og = d_og;
    io.tensor_layout = tensor_layout;
    io.qk_quant_gran = qk_quant_gran;
    io.pv_accum_dtype = pv_accum_dtype;
    io.is_causal = is_causal;
    io.smooth_k = smooth_k;
    io.dtype_mode = dtype_mode;
    io.sm_scale = sm_scale;
    return sage_attn::twin::RunSequential(io);
}

}  // extern "C"

extern "C" {

/**
 * 量化中间态导出（golden 位级比对单测专用）。
 * 必须先调用 SageAttnTwinRunForward；六个出参指针指向静态缓冲，下次 RunForward 前有效。
 * pad_sk_v 出参返回 V 序列维 pad 长度（d_pad 由 q/k/v 各自尺寸隐含）。
 */
void SageAttnTwinQuantDump(const signed char** q8, const signed char** k8,
                           const unsigned char** v8, const float** qs,
                           const float** ks, const float** vs,
                           unsigned* d_pad, unsigned* pad_sk_v) {
    sage_attn::twin::TwinQuantDump dump;
    // RunSequential 的 dump 出参填充自静态缓冲；此处经一次轻量重放不现实，
    // 直接复用头文件约定的静态缓冲导出接口：
    sage_attn::twin::ExportQuantDump(q8, k8, v8, qs, ks, vs, d_pad, pad_sk_v);
}

}  // extern "C"
