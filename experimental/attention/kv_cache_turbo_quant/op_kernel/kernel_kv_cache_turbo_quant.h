/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file kernel_kv_cache_turbo_quant.h
 * \brief Encode Kernel。算法真源：encode_one_core.h（与 golden.py / cpu_encode_twin 同核）。
 * 参考仓 turboquant-ascendC = TQ4 债路径，禁抄打包/接口。
 *
 * 首版功能绿：标量 GetValue/SetValue + EncodeOne 模板加载器（矩阵按元素读 GM，不整表进栈）。
 * TODO(perf)：GM↔UB DataCopy/DataCopyPad + Cast + 向量化点积；删除热路径 Get/SetValue。
 */
#ifndef KERNEL_KV_CACHE_TURBO_QUANT_H_
#define KERNEL_KV_CACHE_TURBO_QUANT_H_

#include "kernel_operator.h"

using namespace AscendC;

#define KV_ENC_INLINE __aicore__ inline
#include "encode_one_core.h"
#undef KV_ENC_INLINE

namespace KvCacheTurboQuantKernel {

constexpr int64_t kHeadDim = KvEncodeCore::kHeadDim;

struct TilingView {
    int64_t numTokens;
    int64_t numKvHeads;
    int64_t mseBits;
    int64_t idxBytes;
    int64_t qjlBytes;
    uint32_t coreNum;
};

/** Device 标量 Sqrt（禁 std::sqrt）。与 tests/verify 中 KvEncSqrtDevice 同公式。 */
struct DeviceSqrt {
    __aicore__ inline float operator()(float x) const
    {
        if (x <= 0.0f) {
            return 0.0f;
        }
        float y = x;
        if (y < 1.0f) {
            y = 1.0f;
        }
        for (int k = 0; k < 8; ++k) {
            y = 0.5f * (y + x / y);
        }
        return y;
    }
};

/** GM 矩阵加载器：rot(j,i)=H[j,i]，避免 128×128 进栈 */
struct GmMatLoad {
    GlobalTensor<float> gm;
    __aicore__ inline float operator()(int j, int i) const
    {
        return gm.GetValue(static_cast<int64_t>(j) * kHeadDim + static_cast<int64_t>(i));
    }
};

class KernelKvCacheTurboQuant {
public:
    __aicore__ inline KernelKvCacheTurboQuant() {}

    __aicore__ inline void Init(GM_ADDR kv, GM_ADDR rotation, GM_ADDR qjl, GM_ADDR quantIdx, GM_ADDR quantQjl,
                                GM_ADDR quantNorm, GM_ADDR quantGamma, const TilingView &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        totalVectors_ = tiling_.numTokens * tiling_.numKvHeads;

        kvGm_.SetGlobalBuffer((__gm__ bfloat16_t *)kv);
        rotGm_.SetGlobalBuffer((__gm__ float *)rotation);
        qjlGm_.SetGlobalBuffer((__gm__ float *)qjl);
        idxGm_.SetGlobalBuffer((__gm__ uint8_t *)quantIdx);
        qjlOutGm_.SetGlobalBuffer((__gm__ uint8_t *)quantQjl);
        normGm_.SetGlobalBuffer((__gm__ bfloat16_t *)quantNorm);
        gammaGm_.SetGlobalBuffer((__gm__ bfloat16_t *)quantGamma);

        pipe_->InitBuffer(xBuf_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(rowBuf_, kHeadDim * sizeof(float));
        pipe_->InitBuffer(idxU8Buf_, kHeadDim * sizeof(uint8_t));
        pipe_->InitBuffer(packBuf_, 64 * sizeof(uint8_t));
        pipe_->InitBuffer(bf16Buf_, kHeadDim * sizeof(bfloat16_t));
    }

    __aicore__ inline void Process()
    {
        const int64_t coreIdx = GetBlockIdx();
        if (coreIdx >= totalVectors_) {
            return;
        }
        for (int64_t v = coreIdx; v < totalVectors_; v += tiling_.coreNum) {
            ProcessOne(v);
        }
    }

private:
    __aicore__ inline float LoadBf16AsFloat(int64_t idx)
    {
        // 禁止 static_cast<float>(bfloat16_t)：bisheng 报 not support bf16 type cast
        return AscendC::ToFloat(kvGm_.GetValue(idx));
    }

    __aicore__ inline void StoreFloatAsBf16(GlobalTensor<bfloat16_t> &dst, int64_t idx, float v)
    {
        // 用 Cast API，禁止 C++ static_cast 到 bfloat16_t
        LocalTensor<float> fLocal = tmpBuf_.Get<float>();
        LocalTensor<bfloat16_t> bLocal = bf16Buf_.Get<bfloat16_t>();
        fLocal.SetValue(0, v);
        Cast(bLocal, fLocal, RoundMode::CAST_RINT, 1);
        dst.SetValue(idx, bLocal.GetValue(0));
    }

    /*!
     * 迁入 EncodeOne：bf16 KV → fp32 → encode_one_core → 打包写回。
     * TODO(perf)：替换 GetValue/SetValue 为 DataCopy / DataCopyPad + Cast。
     */
    __aicore__ inline void ProcessOne(int64_t vecId)
    {
        const int64_t base = vecId * kHeadDim;
        float x[kHeadDim];
        for (int i = 0; i < kHeadDim; ++i) {
            x[i] = LoadBf16AsFloat(base + i);
        }

        const int mseBits = static_cast<int>(tiling_.mseBits);
        uint8_t idxOut[64];
        uint8_t qjlOut[16];
        float norm = 0.f;
        float gamma = 0.f;
        for (int i = 0; i < 64; ++i) {
            idxOut[i] = 0;
        }
        for (int i = 0; i < 16; ++i) {
            qjlOut[i] = 0;
        }

        GmMatLoad rotLoad{rotGm_};
        GmMatLoad qjlLoad{qjlGm_};
        KvEncodeCore::EncodeOne(x, rotLoad, qjlLoad, DeviceSqrt{}, mseBits, idxOut, qjlOut, &norm, &gamma);

        const int64_t idxBytes = tiling_.idxBytes;
        const int64_t qjlBytes = tiling_.qjlBytes;
        const int64_t idxBase = vecId * idxBytes;
        const int64_t qjlBase = vecId * qjlBytes;
        for (int64_t b = 0; b < idxBytes; ++b) {
            idxGm_.SetValue(idxBase + b, idxOut[b]);
        }
        for (int64_t b = 0; b < qjlBytes; ++b) {
            qjlOutGm_.SetValue(qjlBase + b, qjlOut[b]);
        }
        StoreFloatAsBf16(normGm_, vecId, norm);
        StoreFloatAsBf16(gammaGm_, vecId, gamma);

        (void)pipe_;
        (void)xBuf_;
        (void)rowBuf_;
        (void)idxU8Buf_;
        (void)packBuf_;
    }

    TilingView tiling_{};
    TPipe *pipe_{nullptr};
    int64_t totalVectors_{0};

    GlobalTensor<bfloat16_t> kvGm_;
    GlobalTensor<float> rotGm_;
    GlobalTensor<float> qjlGm_;
    GlobalTensor<uint8_t> idxGm_;
    GlobalTensor<uint8_t> qjlOutGm_;
    GlobalTensor<bfloat16_t> normGm_;
    GlobalTensor<bfloat16_t> gammaGm_;

    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> rowBuf_;
    TBuf<TPosition::VECCALC> idxU8Buf_;
    TBuf<TPosition::VECCALC> packBuf_;
    TBuf<TPosition::VECCALC> bf16Buf_;
};

} // namespace KvCacheTurboQuantKernel

#endif // KERNEL_KV_CACHE_TURBO_QUANT_H_
