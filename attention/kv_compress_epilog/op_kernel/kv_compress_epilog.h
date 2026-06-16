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
 * \file kv_compress_epilog.h
 * \brief KV compress epilog kernel implementation
 */

#ifndef KV_COMPRESS_EPILOG_H
#define KV_COMPRESS_EPILOG_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kv_compress_epilog_common.h"

using namespace AscendC;

namespace KvCompressEpilogOps {

constexpr int32_t FOUR_UNFOLD = 4;   // 拷出按 4 行展开

template <typename T0, typename U, typename T1>
class KvCompressEpilog {
public:
    __aicore__ inline KvCompressEpilog(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR slotMapping,
        GM_ADDR kvCache,
        const KvCompressEpilogTilingData* tilingData);

    __aicore__ inline void Process();

    __aicore__ inline void SetMaxValue();

private:
    TPipe* pipe_ = nullptr;

    GlobalTensor<T0> xGm;
    GlobalTensor<U> slotMappingGm;
    GlobalTensor<T1> kvCacheGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> kvCacheQue;
    TBuf<QuePosition::VECCALC> kvCacheScaleBuf;

    LocalTensor<T0> xLocal;
    LocalTensor<T1> kvCacheLocal;
    int64_t validIdx = 0;
    float maxValue = 0.0f;
    float fp8Min = 0.0f;
    float fp8Max = 0.0f;
    float scalesAttr = 1.0f;

    // Tiling data
    const KvCompressEpilogTilingData* tilingData = nullptr;
};

// Template implementations

template <typename T0, typename U, typename T1>
__aicore__ inline void KvCompressEpilog<T0, U, T1>::Init(
    GM_ADDR x,
    GM_ADDR slotMapping,
    GM_ADDR kvCache,
    const KvCompressEpilogTilingData* tilingDataPtr)
{
        tilingData = tilingDataPtr;
        scalesAttr = tilingData->scalesAttr;

        int64_t xGmBaseOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->d;

        xGm.SetGlobalBuffer((__gm__ T0*)x + xGmBaseOffset);
        kvCacheGm.SetGlobalBuffer((__gm__ T1*)kvCache);
        slotMappingGm.SetGlobalBuffer((__gm__ U*)slotMapping);

        pipe_->InitBuffer(xQue, 2, tilingData->rowFactor * RoundUp<T0>(tilingData->d) * sizeof(T0));
        pipe_->InitBuffer(kvCacheQue, 2, tilingData->rowFactor * RoundUp<T1>(tilingData->kvCacheCol) * sizeof(T1));

        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
}

template <typename T0, typename U, typename T1>
 __aicore__ inline void KvCompressEpilog<T0, U, T1>::Process()
    {
        SetMaxValue();
        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                     tilingData->tailRowFactorOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            int64_t xRowBase = rowOuterIdx * tilingData->rowFactor * tilingData->d;   // xGm 已按 block base 偏移
            int64_t slotGmBase = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor;
            // ---- 优化2: 取消输入侧 slot==-1 压缩与逐行同步, 整块连续搬入 x ----
            //   x buffer 行步长 RoundUp<T0>(d)==d (kv 的 d 为 64 倍数), 一条大描述符整块搬入全部行
            xLocal = xQue.template AllocTensor<T0>();
            CopyIn(xGm[xRowBase], xLocal, curRowFactor, tilingData->d);
            xQue.template EnQue(xLocal);
            xLocal = xQue.template DeQue<T0>();

            kvCacheLocal = kvCacheQue.template AllocTensor<T1>();
            if (tilingData->quantMode == QUANT_MODE_GROUP_QUANT_BF16) {
                if (tilingData->roundScale == 1) {
                    VFProcessFP8PerGroupQuant<T0, T1, true, true>(kvCacheLocal, xLocal, maxValue, fp8Min, fp8Max,
                        curRowFactor, tilingData->d, tilingData->concatCol, tilingData->padCol);
                } else {
                    VFProcessFP8PerGroupQuant<T0, T1, false, true>(kvCacheLocal, xLocal, maxValue, fp8Min, fp8Max,
                        curRowFactor, tilingData->d, tilingData->concatCol, tilingData->padCol);
                }
            } else if (tilingData->quantMode == QUANT_MODE_GROUP_QUANT_E8M0) {
                if (tilingData->roundScale == 1) {
                    VFProcessFP8PerGroupQuant<T0, T1, true, false>(kvCacheLocal, xLocal, maxValue, fp8Min, fp8Max,
                        curRowFactor, tilingData->d, tilingData->concatCol, tilingData->padCol);
                } else {
                    VFProcessFP8PerGroupQuant<T0, T1, false, false>(kvCacheLocal, xLocal, maxValue, fp8Min, fp8Max,
                        curRowFactor, tilingData->d, tilingData->concatCol, tilingData->padCol);
                }
            }

            xQue.template FreeTensor(xLocal);

            kvCacheQue.template EnQue(kvCacheLocal);
            kvCacheLocal = kvCacheQue.template DeQue<T1>();
            // ---- 优化1+3: 拷出 4 展开 + 直接从 GM 读 slot (无 UB slot 缓冲); slot==-1 丢弃; 保留分页偏移 ----
            int64_t r = 0;
            for (; r + FOUR_UNFOLD <= curRowFactor; r += FOUR_UNFOLD) {
                for (int32_t k = 0; k < FOUR_UNFOLD; k++) {
                    int64_t slot = static_cast<int64_t>(slotMappingGm.GetValue(slotGmBase + r + k));
                    if (slot == -1) {
                        continue;
                    }
                    int64_t block = slot / tilingData->kvCacheBlockSize;
                    int64_t posInBlock = slot % tilingData->kvCacheBlockSize;
                    int64_t gmOffset =
                        block * tilingData->kvCacheBlockStride + posInBlock * tilingData->kvCacheRowStride;
                    CopyOut(
                        kvCacheLocal[(r + k) * RoundUp<T1>(tilingData->kvCacheCol)],
                        kvCacheGm[gmOffset], 1, tilingData->kvCacheCol);
                }
            }
            for (; r < curRowFactor; r++) {
                int64_t slot = static_cast<int64_t>(slotMappingGm.GetValue(slotGmBase + r));
                if (slot == -1) {
                    continue;
                }
                int64_t block = slot / tilingData->kvCacheBlockSize;
                int64_t posInBlock = slot % tilingData->kvCacheBlockSize;
                int64_t gmOffset =
                    block * tilingData->kvCacheBlockStride + posInBlock * tilingData->kvCacheRowStride;
                CopyOut(
                    kvCacheLocal[r * RoundUp<T1>(tilingData->kvCacheCol)],
                    kvCacheGm[gmOffset], 1, tilingData->kvCacheCol);
            }
            kvCacheQue.template FreeTensor(kvCacheLocal);
        }
    }

    template <typename T0, typename U, typename T1>
    __aicore__ inline void KvCompressEpilog<T0, U, T1>::SetMaxValue()
    {
        maxValue = static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
        fp8Max = FP8_E4M3FN_MAX_VALUE;
        fp8Min = FP8_E4M3FN_MIN_VALUE;
    }

}
#endif  // KV_COMPRESS_EPILOG_H