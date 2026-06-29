/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file lightning_indexer_kl_loss.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_KL_LOSS_H
#define LIGHTNING_INDEXER_KL_LOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lightning_indexer_kl_loss_tiling_data.h"
#include "lightning_indexer_kl_loss_tiling_key.h"

namespace NsLightningIndexerKlLoss {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class LightningIndexerKlLoss {
public:
    __aicore__ inline LightningIndexerKlLoss(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const LightningIndexerKlLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(const int32_t dataLength);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;

    int64_t blockLength_ = 0;
    int64_t tileNum_ = 0;
    uint32_t tileLength_ = 0;
};

template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::Init(GM_ADDR targetScore, GM_ADDR indexProbs, GM_ADDR loss, const LightningIndexerKlLossTilingData* tilingData)
{
    
}

template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::CopyIn(int32_t progress)
{
    
}

template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::CopyOut(int32_t progress)
{
    
}

template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::Compute(int32_t progress)
{
    
}

template <typename T>
__aicore__ inline void LightningIndexerKlLoss<T>::Process()
{
    
}

} // namespace NsLightningIndexerKlLoss
#endif // LIGHTNING_INDEXER_KL_LOSS_H