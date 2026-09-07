/*
 * Copyright (c) 2026 G_W_E. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "softmax.h"
extern "C" __global__ __aicore__ void sage_half_softmax_debug(GM_ADDR raw, GM_ADDR kg, GM_ADDR qg, GM_ADDR values,
                                                              GM_ADDR state, GM_ADDR probabilities, uint32_t blocks,
                                                              uint32_t sk, uint32_t qstart, uint32_t sq,
                                                              uint32_t causal, float scale)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Catlass::Arch::Resource<Sage::Arch> r;
    auto scores = r.ubBuf.GetBufferByByte<half>(0);
    auto ks = r.ubBuf.GetBufferByByte<float>(32768);
    auto qs = r.ubBuf.GetBufferByByte<float>(33792);
    auto m = r.ubBuf.GetBufferByByte<half>(34048);
    auto d = r.ubBuf.GetBufferByByte<float>(34304);
    auto a = r.ubBuf.GetBufferByByte<float>(34560);
    auto p = r.ubBuf.GetBufferByByte<float8_e4m3_t>(34816);
    auto qh = r.ubBuf.GetBufferByByte<half>(51200);
    auto kh = r.ubBuf.GetBufferByByte<half>(51456);
    auto scratch = r.ubBuf.GetBufferByByte<float>(51968);
    GlobalTensor<half> sg, o;
    sg.SetGlobalBuffer((__gm__ half *)raw);
    o.SetGlobalBuffer((__gm__ half *)values);
    GlobalTensor<float> k, q, s;
    k.SetGlobalBuffer((__gm__ float *)kg);
    q.SetGlobalBuffer((__gm__ float *)qg);
    s.SetGlobalBuffer((__gm__ float *)state);
    GlobalTensor<float8_e4m3_t> pg;
    pg.SetGlobalBuffer((__gm__ float8_e4m3_t *)probabilities);
    Duplicate(m, half(-65504), 128);
    Duplicate(d, 0.0f, 64);
    Duplicate(a, 0.0f, 64);
    DataCopy(qs, q, 64);
    for (uint32_t b = 0; b < blocks; ++b) {
        DataCopy(scores, sg[b * 64 * 256], 64 * 256);
        DataCopy(ks, k[b * 256], 256);
        Sage::Sync<HardEvent::MTE2_V>();
        Sage::SoftmaxHalfVF<64, 256, false, true>(
            (__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ float *)ks.GetPhyAddr(), (__ubuf__ float *)qs.GetPhyAddr(),
            (__ubuf__ half *)m.GetPhyAddr(), (__ubuf__ float *)d.GetPhyAddr(), (__ubuf__ float *)a.GetPhyAddr(),
            (__ubuf__ float8_e4m3_t *)p.GetPhyAddr(), (__ubuf__ half *)qh.GetPhyAddr(),
            (__ubuf__ half *)kh.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(), qstart, sq, b * 256, sk, causal,
            scale);
        Cast(scratch, m, RoundMode::CAST_NONE, 64);
        PipeBarrier<PIPE_V>();
        Sage::WriteRow(o, b * 64 * 256, scores, 64 * 256);
        Sage::WriteRow(s, b * 192, scratch, 64);
        Sage::WriteRow(s, b * 192 + 64, d, 64);
        Sage::WriteRow(s, b * 192 + 128, a, 64);
        Sage::WriteRow(pg, b * 64 * 256, p, 64 * 256);
        Sage::Sync<HardEvent::MTE3_MTE2>();
    }
}
extern "C" void sage_half_softmax_debug_launch(void *stream, void **p, uint32_t blocks, uint32_t sk, uint32_t qstart,
                                               uint32_t sq, uint32_t causal, float scale)
{
    sage_half_softmax_debug<<<1, nullptr, stream>>>((uint8_t *)p[0], (uint8_t *)p[1], (uint8_t *)p[2], (uint8_t *)p[3],
                                                    (uint8_t *)p[4], (uint8_t *)p[5], blocks, sk, qstart, sq, causal,
                                                    scale);
}
