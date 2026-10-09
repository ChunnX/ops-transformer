/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file mixed_quant_flash_attn_empty_kernel.h
 * \brief Empty-tensor kernel and sink handling for MixedQuantFlashAttn.
 */

#ifndef MIXED_QUANT_FLASH_ATTN_EMPTY_KERNEL_H_
#define MIXED_QUANT_FLASH_ATTN_EMPTY_KERNEL_H_

#include "kernel_operator.h"
#include "../mixed_quant_flash_attn_tiling_data.h"
#include "../utils/mixed_quant_flash_attn_common_def.h"
#include "mqfa_init_output.h"

using namespace AscendC;

namespace MixedQuantFlashAttnEmpty {

__aicore__ inline uint64_t GetSeqLen(GlobalTensor<int32_t> &seqLenGm, uint32_t size, uint32_t batchIdx,
                                     uint64_t defaultValue)
{
    if (size == 0) {
        return defaultValue;
    }
    return static_cast<uint64_t>(seqLenGm.GetValue(size == 1 ? 0 : batchIdx));
}

template <uint8_t IN_OUT_LAYOUT>
__aicore__ inline void InitEmptyKvSinkLse(
    __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *sinks,
    __gm__ uint8_t *softmaxLse, __tiling_data_ptr__ optiling::MixedQuantFlashAttnTiling *tilingData)
{
    if ASCEND_IS_AIV {
        const auto &baseParams = tilingData->mixedQuantFlashAttnBaseParams;
        if (!baseParams.needInitOutput || !baseParams.hasSink || !baseParams.isSoftMaxLseEnable ||
            (!baseParams.emptyTensor && baseParams.seqUsedKvSize == 0) ||
            (baseParams.emptyTensor && baseParams.totalOutputSize == 0) || baseParams.totalLseSize == 0) {
            return;
        }

        GlobalTensor<int32_t> seqUsedQGm;
        GlobalTensor<int32_t> seqUsedKvGm;
        GlobalTensor<int32_t> cuSeqLensQGm;
        GlobalTensor<float> sinkGm;
        GlobalTensor<float> softmaxLseGm;
        if (baseParams.seqUsedQSize != 0) {
            seqUsedQGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, baseParams.seqUsedQSize);
        }
        if (baseParams.seqUsedKvSize != 0) {
            seqUsedKvGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, baseParams.seqUsedKvSize);
        }
        sinkGm.SetGlobalBuffer((__gm__ float *)sinks);
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        if constexpr (IN_OUT_LAYOUT == InOutLayoutType_TND) {
            cuSeqLensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ, baseParams.cuSeqLensQSize);
        }

        uint32_t n1Size = baseParams.n2Size * baseParams.gSize;
        uint32_t coreIdx = GetBlockIdx();
        uint32_t coreNum = baseParams.coreNum;
        for (uint32_t batchIdx = 0; batchIdx < baseParams.bSize; ++batchIdx) {
            if (GetSeqLen(seqUsedKvGm, baseParams.seqUsedKvSize, batchIdx, baseParams.s2Size) != 0) {
                continue;
            }

            uint64_t qBase = 0;
            uint64_t qSize = baseParams.s1Size;
            if constexpr (IN_OUT_LAYOUT == InOutLayoutType_TND) {
                qBase = static_cast<uint64_t>(cuSeqLensQGm.GetValue(batchIdx));
                qSize = static_cast<uint64_t>(cuSeqLensQGm.GetValue(batchIdx + 1U)) - qBase;
            }
            uint64_t usedQ = GetSeqLen(seqUsedQGm, baseParams.seqUsedQSize, batchIdx, qSize);
            qSize = usedQ < qSize ? usedQ : qSize;
            uint64_t rowCount = qSize * n1Size;
            for (uint64_t row = coreIdx; row < rowCount; row += coreNum) {
                uint32_t headIdx = static_cast<uint32_t>(row / qSize);
                uint64_t tokenIdx = row % qSize;
                uint64_t lseOffset;
                if constexpr (IN_OUT_LAYOUT == InOutLayoutType_TND) {
                    headIdx = static_cast<uint32_t>(row % n1Size);
                    tokenIdx = row / n1Size;
                    lseOffset = (qBase + tokenIdx) * n1Size + headIdx;
                } else {
                    lseOffset = (static_cast<uint64_t>(batchIdx) * n1Size + headIdx) * baseParams.s1Size +
                                tokenIdx;
                }
                softmaxLseGm.SetValue(lseOffset, sinkGm.GetValue(headIdx));
            }
        }
    }
}

template <typename OUT_T, uint8_t IN_OUT_LAYOUT>
class MixedQuantFlashAttnEmptyKernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t *attnOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *cuSeqLensQ,
        __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *sinks,
        __tiling_data_ptr__ optiling::MixedQuantFlashAttnTiling *tilingData)
    {
        attnOut_ = attnOut;
        softmaxLse_ = softmaxLse;
        cuSeqLensQ_ = cuSeqLensQ;
        seqUsedQ_ = seqUsedQ;
        seqUsedKv_ = seqUsedKv;
        sinks_ = sinks;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const auto &baseParams = tilingData_->mixedQuantFlashAttnBaseParams;
            GlobalTensor<OUT_T> attentionOutGm;
            attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attnOut_);
            constexpr uint32_t outputBufferElements = BUFFER_SIZE_BYTE_8K / sizeof(OUT_T);
            AttentionCommon::InitOutput<OUT_T, EVENT_ID0, 0, outputBufferElements>(
                attentionOutGm, baseParams.totalOutputSize, baseParams.coreNum, static_cast<OUT_T>(0));

            if (!baseParams.isSoftMaxLseEnable || baseParams.totalLseSize == 0) {
                return;
            }

            GlobalTensor<float> softmaxLseGm;
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse_);
            constexpr uint32_t lseBufferElements = BUFFER_SIZE_BYTE_8K / sizeof(float);
            AttentionCommon::InitOutput<float, EVENT_ID1, BUFFER_SIZE_BYTE_8K, lseBufferElements>(
                softmaxLseGm, baseParams.totalLseSize, baseParams.coreNum, static_cast<float>(3e+99));
            SyncAll();
            InitEmptyKvSinkLse<IN_OUT_LAYOUT>(cuSeqLensQ_, seqUsedQ_, seqUsedKv_, sinks_, softmaxLse_, tilingData_);
        }
    }

private:
    __gm__ uint8_t *attnOut_ = nullptr;
    __gm__ uint8_t *softmaxLse_ = nullptr;
    __gm__ uint8_t *cuSeqLensQ_ = nullptr;
    __gm__ uint8_t *seqUsedQ_ = nullptr;
    __gm__ uint8_t *seqUsedKv_ = nullptr;
    __gm__ uint8_t *sinks_ = nullptr;
    __tiling_data_ptr__ optiling::MixedQuantFlashAttnTiling *tilingData_ = nullptr;
};

} // namespace MixedQuantFlashAttnEmpty

#endif // MIXED_QUANT_FLASH_ATTN_EMPTY_KERNEL_H_
